"""
CHITIN · src/engine.py
──────────────────────
Core engine for localized manifold subtraction.

CHITIN v2 upgrades over v1
──────────────────────────
1. AUTO-CALIBRATION (Pareto sweep)
   fit() now runs a sweep over (k, n_pcs, distance_metric) before committing
   to any single operating point. For each combination it evaluates three
   objectives on a subsample of the training data:
     - Rank disruption   : 1 - mean_spearman_rho  (maximise)
     - Discrimination    : post/pre pairwise cosine distance ratio  (maximise ≥ 1 ideally, but we want to preserve as much as possible)
     - Signal stability  : 1 / std(delta_norms)  (maximise — low std = clean correction)
   The Pareto front is computed across these three objectives and the
   operating point with the best rank disruption subject to signal stability
   ≥ 0.5× the best stable point is selected automatically.
   The user can disable this (auto_calibrate: false in yaml) and manually
   specify k and n_pcs as before.

2. PC DECOMPOSITION CORRECTION (optional, complement to KNN)
   If correction_mode: 'pc_decomposition' or 'hybrid' in yaml, CHITIN
   identifies which PCs encode systematic variation by computing cosine
   similarity between each PC loading vector and the systematic variation
   vector V (O_pert - C_ctrl). PCs with cosine similarity above a threshold
   are projected out of every perturbed cell's expression vector.
   This is more effective than KNN for homogeneous cell lines (iPSC, hESC)
   where control manifold coverage is sparse.
   Modes:
     'knn'            : original CHITIN v1 KNN subtraction (default)
     'pc_decomposition': project out systematic PCs, no KNN
     'hybrid'         : apply PC decomposition first, then KNN on residuals

Architecture:
  fit()       → Auto-calibrate OR use manual params → fit KNN + PC decomp
  transform() → Apply correction using fitted model
  sweep()     → Pareto sweep over (k, n_pcs, distance_metric) — called by fit()
"""

import gc
import warnings
from itertools import product

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from .utils import log_phase, snapshot, log_memory, force_gc

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
#  DEFAULT SWEEP GRIDS
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_K_GRID      = [1, 3, 5, 10, 15, 20, 30, 50]
DEFAULT_NPCS_GRID   = [5, 10, 15, 20, 30, 50]
DEFAULT_METRIC_GRID = ["euclidean", "cosine"]

# Subsample size for fast Pareto sweep evaluation
SWEEP_SUBSAMPLE_CELLS  = 2000
SWEEP_SUBSAMPLE_GENES  = 300
SWEEP_N_PAIRS          = 500   # pairs for pairwise discrimination in sweep
SWEEP_N_GENES_RHO      = 100   # genes for rank disruption in sweep

# PC decomposition: cosine similarity threshold above which a PC is
# classified as encoding systematic variation
PC_SYSTEMATIC_COS_THRESHOLD = 0.3


# ═══════════════════════════════════════════════════════════════════════════
#  CHITIN MODEL v2
# ═══════════════════════════════════════════════════════════════════════════

class ChitinModel:
    """
    Fitted CHITIN model — v2 with auto-calibration and PC decomposition.

    Usage (auto-calibrate mode, recommended):
        model = ChitinModel()
        model.fit(adata_train, cfg, logger)   # sweeps k/n_pcs automatically
        delta_train = model.transform(adata_train, cfg, logger)
        delta_val   = model.transform(adata_val,   cfg, logger)
        delta_test  = model.transform(adata_test,  cfg, logger)

    Usage (manual mode — set auto_calibrate: false in yaml):
        Same as above, uses cfg['knn']['k'] and cfg['knn']['n_pcs'] directly.
    """

    def __init__(self):
        # KNN state
        self.pca_components   = None   # (n_genes × n_pcs) PCA loadings
        self.nn_model         = None   # fitted NearestNeighbors on ctrl PCA
        self.X_ctrl_expr      = None   # (n_ctrl × n_genes) control expression
        self.k                = None
        self.n_pcs            = None
        self.distance_metric  = None

        # PC decomposition state
        self.systematic_pc_indices  = []    # which PCs to project out
        self.systematic_pc_loadings = None  # (n_systematic × n_genes)
        self.V_systematic_cos       = None  # cosine similarities per PC

        # Sweep results
        self.sweep_results    = None   # DataFrame of all sweep combinations
        self.pareto_front     = None   # DataFrame of Pareto-optimal points
        self.selected_params  = None   # dict: {'k', 'n_pcs', 'metric'}

        # Correction mode
        self.correction_mode  = "knn"  # 'knn' | 'pc_decomposition' | 'hybrid'

        self.is_fitted = False

    # ─────────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────────

    def fit(self, adata, cfg: dict, logger):
        """
        Fit CHITIN on a training split.

        If cfg['calibration']['auto_calibrate'] is True (default), runs the
        Pareto sweep to select optimal (k, n_pcs, distance_metric) before fitting.
        Otherwise uses cfg['knn']['k'], cfg['knn']['n_pcs'], cfg['knn']['distance_metric'].
        """
        log_phase(logger, "CHITIN v2 · Fitting Reference Manifold")

        pert_col   = cfg["dataset"]["perturbation_col"]
        ctrl_label = cfg["dataset"]["control_label"]

        self.correction_mode = cfg.get("correction", {}).get(
            "mode", "knn")
        auto_calibrate = cfg.get("calibration", {}).get(
            "auto_calibrate", True)

        labels     = adata.obs[pert_col].values
        ctrl_mask  = labels == ctrl_label
        ctrl_idx   = np.where(ctrl_mask)[0]
        pert_idx   = np.where(~ctrl_mask)[0]
        n_ctrl     = len(ctrl_idx)
        n_genes    = adata.n_vars

        logger.info(f"  Controls: {n_ctrl:,}  |  Perturbed: {len(pert_idx):,}  "
                    f"|  Genes: {n_genes:,}  |  Mode: {self.correction_mode}")

        if n_ctrl == 0:
            raise ValueError("No control metacells found.")

        # ── Auto-calibrate: Pareto sweep ──────────────────────────────────
        if auto_calibrate:
            logger.info("  AUTO-CALIBRATE: running Pareto sweep...")
            k_grid    = cfg.get("calibration", {}).get("k_grid",    DEFAULT_K_GRID)
            npcs_grid = cfg.get("calibration", {}).get("n_pcs_grid", DEFAULT_NPCS_GRID)
            metric_grid = cfg.get("calibration", {}).get(
                "metric_grid", DEFAULT_METRIC_GRID)

            # Clip k_grid to n_ctrl - 1
            k_grid = [k for k in k_grid if k < n_ctrl]
            if not k_grid:
                k_grid = [max(1, n_ctrl - 1)]

            self.sweep_results, self.pareto_front, self.selected_params = \
                self._pareto_sweep(adata, cfg, logger,
                                   k_grid, npcs_grid, metric_grid,
                                   ctrl_idx, pert_idx)

            self.k               = self.selected_params["k"]
            self.n_pcs           = self.selected_params["n_pcs"]
            self.distance_metric = self.selected_params["metric"]

            logger.info(f"  ★ Selected: k={self.k}, n_pcs={self.n_pcs}, "
                        f"metric={self.distance_metric}")
        else:
            self.k               = cfg["knn"]["k"]
            self.n_pcs           = cfg["knn"]["n_pcs"]
            self.distance_metric = cfg["knn"]["distance_metric"]
            logger.info(f"  Manual params: k={self.k}, n_pcs={self.n_pcs}, "
                        f"metric={self.distance_metric}")

        solver = cfg["knn"]["svd_solver"]

        # ── Compute PCA with selected n_pcs ───────────────────────────────
        logger.info(f"  Computing PCA ({self.n_pcs} PCs, solver={solver})...")
        adata_work = adata.copy()
        sc.pp.scale(adata_work, max_value=10, zero_center=False)
        sc.pp.pca(adata_work, n_comps=self.n_pcs, use_highly_variable=False,
                  svd_solver=solver, zero_center=False)

        adata.obsm["X_pca_chitin"] = adata_work.obsm["X_pca"].copy()
        if "pca" in adata_work.uns:
            adata.uns["chitin_pca_variance_ratio"] = \
                adata_work.uns["pca"]["variance_ratio"]

        if hasattr(adata_work, "varm") and "PCs" in adata_work.varm:
            self.pca_components = adata_work.varm["PCs"].copy()  # (n_genes × n_pcs)

        del adata_work
        force_gc(logger)

        # ── PC decomposition: identify systematic PCs ─────────────────────
        if self.correction_mode in ("pc_decomposition", "hybrid"):
            logger.info("  Identifying systematic PCs via V-alignment...")
            self._fit_pc_decomposition(adata, cfg, logger, ctrl_idx, pert_idx)

        # ── KNN on control PCA coords ─────────────────────────────────────
        if self.correction_mode in ("knn", "hybrid"):
            pca_ctrl = adata.obsm["X_pca_chitin"][ctrl_idx]
            k_actual = min(self.k, n_ctrl - 1)
            if k_actual != self.k:
                logger.warning(f"  k capped to {k_actual} (n_ctrl={n_ctrl})")
                self.k = k_actual

            logger.info(f"  Fitting KNN (k={self.k}, metric={self.distance_metric}) "
                        f"on {n_ctrl:,} controls...")
            self.nn_model = NearestNeighbors(
                n_neighbors=self.k,
                metric=self.distance_metric,
                n_jobs=-1)
            self.nn_model.fit(pca_ctrl)

        # Store control expression for KNN baseline computation
        X = adata.X
        if sp.issparse(X):
            self.X_ctrl_expr = X[ctrl_idx].toarray().astype(np.float32)
        else:
            self.X_ctrl_expr = np.array(X[ctrl_idx], dtype=np.float32)

        self.is_fitted = True
        logger.info("  CHITIN v2 model fitted.")
        log_memory(logger, "post fit")
        return adata

    # ─────────────────────────────────────────────────────────────────────
    # PC DECOMPOSITION FIT
    # ─────────────────────────────────────────────────────────────────────

    def _fit_pc_decomposition(self, adata, cfg, logger, ctrl_idx, pert_idx):
        """
        Identify which PCs encode systematic variation by comparing their
        loading vectors against the systematic variation vector V.

        V = O_pert - C_ctrl  (the average perturbation shift in gene space)

        A PC is classified as systematic if:
          cosine_similarity(PC_loading, V) > PC_SYSTEMATIC_COS_THRESHOLD

        The PC loading vectors are normalised before comparison.
        """
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)

        C_ctrl = X[ctrl_idx].mean(axis=0)
        O_pert = X[pert_idx].mean(axis=0)
        V = O_pert - C_ctrl
        V_norm = np.linalg.norm(V)

        if V_norm < 1e-10:
            logger.warning("  PC decomp: V is near-zero, no systematic PCs identified")
            self.systematic_pc_indices  = []
            self.systematic_pc_loadings = np.zeros((0, X.shape[1]), dtype=np.float32)
            return

        V_unit = V / V_norm

        if self.pca_components is None:
            logger.warning("  PC decomp: no PCA loadings stored, skipping")
            return

        # pca_components is (n_genes × n_pcs), each column is a PC loading
        n_pcs = self.pca_components.shape[1]
        cos_sims = np.zeros(n_pcs)
        for i in range(n_pcs):
            pc = self.pca_components[:, i]
            pc_norm = np.linalg.norm(pc)
            if pc_norm > 1e-10:
                cos_sims[i] = abs(float(np.dot(pc / pc_norm, V_unit)))

        self.V_systematic_cos = cos_sims

        threshold = cfg.get("correction", {}).get(
            "pc_systematic_threshold", PC_SYSTEMATIC_COS_THRESHOLD)

        systematic = np.where(cos_sims > threshold)[0]

        # Also enforce a minimum: always subtract at least the top-1 PC
        # if it has cos > 0.2, regardless of threshold
        if len(systematic) == 0 and cos_sims.max() > 0.2:
            systematic = np.array([np.argmax(cos_sims)])

        self.systematic_pc_indices = systematic.tolist()
        if len(systematic) > 0:
            # Store as (n_systematic × n_genes) unit vectors
            loadings = self.pca_components[:, systematic].T.astype(np.float32)
            norms = np.linalg.norm(loadings, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            self.systematic_pc_loadings = loadings / norms
        else:
            self.systematic_pc_loadings = np.zeros((0, X.shape[1]),
                                                   dtype=np.float32)

        logger.info(f"  Systematic PCs identified: {self.systematic_pc_indices} "
                    f"(cos_sim threshold={threshold:.2f})")
        for i in systematic:
            logger.info(f"    PC{i+1}: cos_sim={cos_sims[i]:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # PARETO SWEEP
    # ─────────────────────────────────────────────────────────────────────

    def _pareto_sweep(self, adata, cfg, logger,
                      k_grid, npcs_grid, metric_grid,
                      ctrl_idx, pert_idx):
        """
        Sweep over (k, n_pcs, distance_metric) combinations.
        For each combination, evaluate 3 objectives on a subsample:
          1. rank_disruption  = 1 - mean_spearman_rho  (higher = better)
          2. disc_ratio       = mean_post_dist / mean_pre_dist  (closer to 1 = better)
                                We want to PRESERVE as much discrimination as possible,
                                so we score as abs(disc_ratio - 1.0), lower = better.
          3. signal_stability = -std(delta_norms) normalised  (lower std = better)

        Returns (results_df, pareto_df, selected_params_dict)
        """
        pert_col   = cfg["dataset"]["perturbation_col"]
        ctrl_label = cfg["dataset"]["control_label"]
        solver     = cfg["knn"]["svd_solver"]

        # Subsample for speed
        rng = np.random.default_rng(42)
        n_pert_sub = min(SWEEP_SUBSAMPLE_CELLS, len(pert_idx))
        pert_sub   = rng.choice(pert_idx, size=n_pert_sub, replace=False)
        n_gene_sub = min(SWEEP_SUBSAMPLE_GENES, adata.n_vars)
        gene_sub   = rng.choice(adata.n_vars, size=n_gene_sub, replace=False)

        X_full = adata.X
        if sp.issparse(X_full):
            X_full = X_full.toarray()
        X_full = X_full.astype(np.float32)

        X_ctrl_full = X_full[ctrl_idx]
        X_pert_sub  = X_full[pert_sub]
        X_pert_gene_sub = X_pert_sub[:, gene_sub]

        # Pre-CHITIN pairwise distances (subsample of pairs)
        pre_dists = _fast_pairwise_cosine(
            X_pert_sub, cfg, pert_col, ctrl_label, adata, pert_sub,
            SWEEP_N_PAIRS, rng)

        n_total = len(k_grid) * len(npcs_grid) * len(metric_grid)
        logger.info(f"  Sweep: {len(k_grid)} k × {len(npcs_grid)} n_pcs × "
                    f"{len(metric_grid)} metrics = {n_total} combinations")

        records = []
        best_pca_cache = {}  # cache PCA by n_pcs to avoid recomputing

        for n_pcs in npcs_grid:
            # Cache PCA computation per n_pcs value
            if n_pcs not in best_pca_cache:
                adata_w = adata.copy()
                sc.pp.scale(adata_w, max_value=10, zero_center=False)
                sc.pp.pca(adata_w, n_comps=n_pcs, use_highly_variable=False,
                          svd_solver=solver, zero_center=False)
                pca_all  = adata_w.obsm["X_pca"].copy()
                pca_loadings = adata_w.varm["PCs"].copy() \
                    if "PCs" in adata_w.varm else None
                del adata_w
                best_pca_cache[n_pcs] = (pca_all, pca_loadings)
            else:
                pca_all, pca_loadings = best_pca_cache[n_pcs]

            pca_ctrl_all = pca_all[ctrl_idx]
            pca_pert_sub = pca_all[pert_sub]

            for metric in metric_grid:
                # Build KNN once per (n_pcs, metric)
                nn = NearestNeighbors(
                    n_neighbors=max(k_grid), metric=metric, n_jobs=-1)
                nn.fit(pca_ctrl_all)
                distances_all, nbr_idx_all = nn.kneighbors(pca_pert_sub)

                for k in k_grid:
                    if k >= len(ctrl_idx):
                        continue
                    try:
                        rec = self._eval_sweep_point(
                            k, n_pcs, metric,
                            X_ctrl_full, X_pert_sub, X_pert_gene_sub,
                            gene_sub, nbr_idx_all, distances_all,
                            pre_dists, rng)
                        records.append(rec)
                    except Exception as e:
                        logger.warning(f"    Sweep k={k} n_pcs={n_pcs} "
                                       f"metric={metric} FAILED: {e}")

        del best_pca_cache
        gc.collect()

        results_df = pd.DataFrame(records)
        if len(results_df) == 0:
            logger.warning("  Sweep produced no valid results, using defaults")
            return results_df, results_df, {
                "k": cfg["knn"]["k"],
                "n_pcs": cfg["knn"]["n_pcs"],
                "metric": cfg["knn"]["distance_metric"]}

        logger.info(f"  Sweep complete: {len(results_df)} valid points")

        # Log top 5 by rank disruption
        top5 = results_df.nlargest(5, "rank_disruption")
        logger.info("  Top 5 by rank disruption:")
        for _, row in top5.iterrows():
            logger.info(f"    k={int(row['k']):<3} n_pcs={int(row['n_pcs']):<3} "
                        f"metric={row['metric']:<10} "
                        f"disruption={row['rank_disruption']:.4f}  "
                        f"disc_ratio={row['disc_ratio']:.4f}  "
                        f"stability={row['signal_stability']:.4f}")

        pareto_df = _compute_pareto_front(results_df)
        logger.info(f"  Pareto front: {len(pareto_df)} points")

        selected = _select_from_pareto(pareto_df, results_df, cfg, logger)
        return results_df, pareto_df, selected

    def _eval_sweep_point(self, k, n_pcs, metric,
                          X_ctrl, X_pert, X_pert_genes,
                          gene_sub, nbr_idx_all, dist_all,
                          pre_dists, rng):
        """Evaluate one (k, n_pcs, metric) combination on the subsample."""
        n_pert = len(X_pert)

        # Compute localized baselines using top-k neighbours
        N_i = np.zeros((n_pert, X_ctrl.shape[1]), dtype=np.float32)
        for i in range(n_pert):
            N_i[i] = X_ctrl[nbr_idx_all[i, :k]].mean(axis=0)

        delta = X_pert - N_i

        # ── Rank disruption (on gene subsample) ───────────────────────────
        delta_genes = delta[:, gene_sub]
        X_pert_genes_arr = X_pert_genes

        rhos = []
        for gi in range(len(gene_sub)):
            pre_vals  = X_pert_genes_arr[:, gi]
            post_vals = delta_genes[:, gi]
            if np.std(pre_vals) < 1e-10 or np.std(post_vals) < 1e-10:
                continue
            r, _ = spearmanr(pre_vals, post_vals)
            if np.isfinite(r):
                rhos.append(r)
        mean_rho  = float(np.mean(rhos)) if rhos else 1.0
        disruption = 1.0 - mean_rho

        # ── Signal stability (std of delta norms) ─────────────────────────
        delta_norms   = np.linalg.norm(delta, axis=1)
        signal_stab   = float(1.0 / (np.std(delta_norms) + 1e-6))

        # ── Pairwise discrimination ratio ─────────────────────────────────
        # Compute post-CHITIN pairwise distances on the subsample
        n_pairs = min(SWEEP_N_PAIRS, len(pre_dists))
        post_d  = []
        pert_labels = np.arange(n_pert)  # use index as proxy perturbation id
        if n_pairs > 0:
            sample_pairs = rng.integers(0, n_pert, size=(n_pairs * 2, 2))
            for pa, pb in sample_pairs[:n_pairs]:
                if pa == pb:
                    continue
                ca = delta[pa]
                cb = delta[pb]
                na = np.linalg.norm(ca)
                nb = np.linalg.norm(cb)
                if na > 1e-12 and nb > 1e-12:
                    cos_d = 1.0 - float(np.dot(ca / na, cb / nb))
                    post_d.append(cos_d)

        pre_mean  = float(np.mean(pre_dists)) if len(pre_dists) > 0 else 1.0
        post_mean = float(np.mean(post_d))    if len(post_d)  > 0 else pre_mean
        disc_ratio = post_mean / (pre_mean + 1e-10)

        return {
            "k":                  k,
            "n_pcs":              n_pcs,
            "metric":             metric,
            "rank_disruption":    disruption,
            "mean_rho":           mean_rho,
            "disc_ratio":         disc_ratio,
            "signal_stability":   signal_stab,
            "delta_norm_mean":    float(delta_norms.mean()),
            "delta_norm_std":     float(delta_norms.std()),
        }

    # ─────────────────────────────────────────────────────────────────────
    # TRANSFORM
    # ─────────────────────────────────────────────────────────────────────

    def transform(self, adata, cfg: dict, logger, label: str = ""):
        """
        Transform any split using the fitted CHITIN v2 model.
        Applies the correction_mode selected during fit().
        """
        if not self.is_fitted:
            raise RuntimeError("CHITIN model not fitted. Call fit() first.")

        tag = f" [{label}]" if label else ""
        log_phase(logger, f"CHITIN v2 · Transforming{tag}")

        pert_col   = cfg["dataset"]["perturbation_col"]
        ctrl_label = cfg["dataset"]["control_label"]

        labels    = adata.obs[pert_col].values
        ctrl_mask = labels == ctrl_label
        pert_mask = ~ctrl_mask
        ctrl_indices = np.where(ctrl_mask)[0]
        pert_indices = np.where(pert_mask)[0]

        logger.info(f"{tag} Controls: {len(ctrl_indices):,}  "
                    f"Perturbed: {len(pert_indices):,}  "
                    f"Mode: {self.correction_mode}")

        if len(pert_indices) == 0:
            logger.warning(f"{tag} No perturbed cells — returning unchanged.")
            return adata

        # Project into PCA space
        adata = self._project_to_pca(adata, cfg, logger)

        # Get expression
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)

        n_genes = adata.n_vars
        n_pert  = len(pert_indices)
        X_pert  = X[pert_indices]

        # ── Apply correction ──────────────────────────────────────────────
        if self.correction_mode == "knn":
            delta_pert = self._apply_knn_correction(
                adata, pert_indices, X_pert, logger, tag)

        elif self.correction_mode == "pc_decomposition":
            delta_pert = self._apply_pc_correction(X_pert, logger, tag)

        elif self.correction_mode == "hybrid":
            # PC decomposition first, then KNN on residuals
            residual = self._apply_pc_correction(X_pert, logger, tag)
            delta_pert = self._apply_knn_correction(
                adata, pert_indices, residual, logger, tag,
                use_delta_as_input=True)
        else:
            raise ValueError(f"Unknown correction_mode: {self.correction_mode}")

        # ── Build output AnnData ──────────────────────────────────────────
        X_delta = np.zeros((adata.n_obs, n_genes), dtype=np.float32)
        X_delta[pert_indices] = delta_pert

        adata_delta = ad.AnnData(
            X=X_delta,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy() if adata.uns else {},
        )
        if adata.obsm:
            for key, val in adata.obsm.items():
                adata_delta.obsm[key] = val.copy()

        # Diagnostic layers
        X_basal = np.zeros_like(X_delta)
        if len(ctrl_indices) > 0:
            X_basal[ctrl_indices] = X[ctrl_indices]
        adata_delta.layers["pre_chitin"] = X.copy()
        adata_delta.layers["basal"]      = X_basal

        adata_delta.obs["chitin_transformed"] = False
        adata_delta.obs.iloc[
            pert_indices,
            adata_delta.obs.columns.get_loc("chitin_transformed")
        ] = True

        # Store calibration info in uns
        adata_delta.uns["chitin_params"] = {
            "k":               self.k,
            "n_pcs":           self.n_pcs,
            "metric":          self.distance_metric,
            "correction_mode": self.correction_mode,
            "systematic_pcs":  self.systematic_pc_indices,
        }

        snapshot(adata_delta, f"Post CHITIN{tag}", logger)
        return adata_delta

    def _apply_knn_correction(self, adata, pert_indices, X_pert,
                               logger, tag, use_delta_as_input=False):
        """KNN-based localized subtraction."""
        pca_pert = adata.obsm["X_pca_chitin"][pert_indices]
        distances, nbr_idx = self.nn_model.kneighbors(pca_pert)
        logger.info(f"{tag} KNN: {len(pert_indices):,} × {self.k} neighbors")

        n_pert  = len(pert_indices)
        n_genes = X_pert.shape[1]
        N_i = np.zeros((n_pert, n_genes), dtype=np.float32)
        for i in range(n_pert):
            N_i[i] = self.X_ctrl_expr[nbr_idx[i]].mean(axis=0)

        return X_pert - N_i

    def _apply_pc_correction(self, X_pert, logger, tag):
        """
        Project out systematic PC components from perturbed expression vectors.

        For each systematic PC (unit loading vector l):
            x_corrected = x - (x · l) * l

        Applied iteratively for each systematic PC.
        """
        if len(self.systematic_pc_indices) == 0:
            logger.warning(f"{tag} PC decomp: no systematic PCs, returning unchanged")
            return X_pert.copy()

        X_corr = X_pert.copy()
        for i, pc_idx in enumerate(self.systematic_pc_indices):
            l = self.systematic_pc_loadings[i]  # (n_genes,) unit vector
            # project out: x = x - (x·l)*l
            projections = X_corr @ l              # (n_pert,)
            X_corr = X_corr - np.outer(projections, l)

        logger.info(f"{tag} PC decomp: projected out "
                    f"{len(self.systematic_pc_indices)} systematic PC(s): "
                    f"{[f'PC{i+1}' for i in self.systematic_pc_indices]}")
        return X_corr

    def _project_to_pca(self, adata, cfg, logger):
        """Project into the training PCA space (or compute fresh if needed)."""
        if self.pca_components is not None:
            X = adata.X
            if sp.issparse(X):
                X = X.toarray()
            X_scaled = np.clip(X.astype(np.float32), None, 10)
            adata.obsm["X_pca_chitin"] = X_scaled @ self.pca_components
        else:
            logger.info("  No stored PCA — computing fresh (standalone mode)...")
            n_pcs  = self.n_pcs or cfg["knn"]["n_pcs"]
            solver = cfg["knn"]["svd_solver"]
            adata_work = adata.copy()
            sc.pp.scale(adata_work, max_value=10, zero_center=False)
            sc.pp.pca(adata_work, n_comps=n_pcs, use_highly_variable=False,
                      svd_solver=solver, zero_center=False)
            adata.obsm["X_pca_chitin"] = adata_work.obsm["X_pca"].copy()
            del adata_work
            force_gc(logger)
        return adata


# ═══════════════════════════════════════════════════════════════════════════
#  PARETO FRONT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _fast_pairwise_cosine(X_pert, cfg, pert_col, ctrl_label,
                           adata, pert_idx, n_pairs, rng):
    """
    Compute a fast subsample of pairwise cosine distances between
    perturbed cells (for sweep baseline comparison).
    """
    labels     = adata.obs[pert_col].values[pert_idx]
    unique_p   = np.unique(labels)
    if len(unique_p) < 2:
        return [1.0] * n_pairs

    centroids = {}
    for p in unique_p:
        m = labels == p
        if m.sum() > 0:
            centroids[p] = X_pert[m].mean(axis=0)

    perts = list(centroids.keys())
    dists = []
    attempts = 0
    while len(dists) < n_pairs and attempts < n_pairs * 10:
        attempts += 1
        i, j = rng.integers(0, len(perts), size=2)
        if i == j:
            continue
        ca, cb = centroids[perts[i]], centroids[perts[j]]
        na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
        if na > 1e-12 and nb > 1e-12:
            dists.append(1.0 - float(np.dot(ca / na, cb / nb)))
    return dists


def _compute_pareto_front(df):
    """
    Identify Pareto-optimal points across three objectives:
      - rank_disruption  (maximise)
      - disc_ratio       (maximise — higher = perturbations more separable post)
      - signal_stability (maximise)

    A point is Pareto-dominated if another point is better on ALL three.
    """
    vals = df[["rank_disruption", "disc_ratio", "signal_stability"]].values
    n    = len(vals)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j >= i on all and > i on at least one
            if (np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i])):
                dominated[i] = True
                break

    return df[~dominated].copy()


def _select_from_pareto(pareto_df, all_df, cfg, logger):
    """
    From the Pareto front, select the operating point that:
      1. Maximises rank disruption (primary objective)
      2. Subject to signal_stability >= 0.5 × max_stability on front
      3. Among those, picks the one with best disc_ratio

    Falls back to the globally best rank_disruption point if no
    stability-filtered point exists.
    """
    min_stability = cfg.get("calibration", {}).get(
        "min_stability_fraction", 0.5)

    max_stab   = pareto_df["signal_stability"].max()
    stab_floor = min_stability * max_stab

    candidates = pareto_df[pareto_df["signal_stability"] >= stab_floor]
    if len(candidates) == 0:
        candidates = pareto_df

    # Among candidates, maximise rank_disruption, break ties on disc_ratio
    candidates = candidates.sort_values(
        ["rank_disruption", "disc_ratio"], ascending=[False, False])
    best = candidates.iloc[0]

    logger.info(f"  Pareto selection: k={int(best['k'])}, "
                f"n_pcs={int(best['n_pcs'])}, metric={best['metric']}")
    logger.info(f"    rank_disruption={best['rank_disruption']:.4f}  "
                f"disc_ratio={best['disc_ratio']:.4f}  "
                f"stability={best['signal_stability']:.4f}")

    return {
        "k":      int(best["k"]),
        "n_pcs":  int(best["n_pcs"]),
        "metric": best["metric"],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS (unchanged API from v1)
# ═══════════════════════════════════════════════════════════════════════════

def fit_and_transform_all(adata_train, adata_val, adata_test, cfg, logger):
    """Fit on train, transform all three splits."""
    model = ChitinModel()
    adata_train  = model.fit(adata_train, cfg, logger)
    delta_train  = model.transform(adata_train, cfg, logger, label="Train")
    delta_val    = model.transform(adata_val,   cfg, logger, label="Val")
    delta_test   = model.transform(adata_test,  cfg, logger, label="Test")
    return model, delta_train, delta_val, delta_test


def run_chitin_standalone(adata, cfg, logger):
    """Standalone mode: fit AND transform on the same dataset."""
    log_phase(logger, "CHITIN v2 · Standalone Mode")

    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    if pert_col not in adata.obs.columns:
        raise ValueError(f"Column '{pert_col}' not in .obs. "
                         f"Available: {list(adata.obs.columns)}")

    n_ctrl = (adata.obs[pert_col].values == ctrl_label).sum()
    if n_ctrl == 0:
        raise ValueError(f"No cells with label '{ctrl_label}' found.")

    model  = ChitinModel()
    adata  = model.fit(adata, cfg, logger)
    delta  = model.transform(adata, cfg, logger, label="Standalone")
    return delta, model
