"""
CHITIN · src/engine.py
──────────────────────
Core Engine: Counterfactual Manifold Subtraction & Causal Isolation

This module isolates the pure, causal transcriptional effect of genetic 
perturbations by mathematically subtracting the ambient biological manifold.

Architectural Foundations
─────────────────────────
1. Localized Counterfactual Inference (KNN Mode): 
   Rather than subtracting a global control mean (which falsely assumes 
   cells exist in a single static state), CHITIN maps perturbed cells to 
   their k-nearest control neighbors in PCA space. This estimates a dynamic 
   counterfactual baseline for every individual cell.

2. Orthogonal PC Decomposition:
   If a perturbation forces a shift along a known physiological axis, 
   CHITIN calculates the cosine similarity between the perturbation shift 
   vector (V) and the principal component (PC) loadings. Systematic PCs are 
   removed via orthogonal projection [ X_corr = X - (X·L)L ], leaving only 
   the novel perturbation signal.

3. Pareto-Optimal Auto-Calibration:
   Hyperparameters (k, n_pcs, metric) are not guessed; they are derived. 
   CHITIN executes a Pareto sweep across the parameter grid, calculating a 
   multivariate frontier that balances Rank Disruption (Spearman), Topology 
   Discrimination (Cosine), and Signal Stability (Norm Variance). It 
   autonomously selects the operating point closest to the Utopian ideal.
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
from sklearn.decomposition import PCA, TruncatedSVD
from joblib import Parallel, delayed

from .utils import log_phase, snapshot, log_memory, force_gc

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
#  DEFAULT SWEEP GRIDS
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_K_GRID      = [1, 2, 3, 4, 5, 10, 15, 20]
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
        self.pca_mean         = None   # (n_genes,) Training mean for geometric alignment
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
            
            # Lock in the winning algorithm if we were exploring
            if self.correction_mode == "explore":
                self.correction_mode = self.selected_params.get("mode", "knn")
                logger.info(f"  ★ Explore Mode Winner: {self.correction_mode.upper()}")

            logger.info(f"  ★ Selected: k={self.k}, n_pcs={self.n_pcs}, "
                        f"metric={self.distance_metric}")
        else:
            self.k               = cfg["knn"]["k"]
            self.n_pcs           = cfg["knn"]["n_pcs"]
            self.distance_metric = cfg["knn"]["distance_metric"]
            logger.info(f"  Manual params: k={self.k}, n_pcs={self.n_pcs}, "
                        f"metric={self.distance_metric}")

        solver = cfg["knn"]["svd_solver"]

        # ── Compute PCA with exact Training Mean Alignment ────────────────
        logger.info(f"  Computing PCA ({self.n_pcs} PCs, solver={solver})...")
        
        # 1. Extract raw training matrix and compute training mean
        X_train = adata.X
        if sp.issparse(X_train):
            X_train = X_train.toarray()
        X_train = X_train.astype(np.float32)
        
        self.pca_mean = X_train.mean(axis=0)
        
        # 2. Mean-center the training data explicitly
        X_centered = X_train - self.pca_mean
        
        # 3. Fit PCA natively using sklearn to guarantee exact V.T extraction
        from sklearn.decomposition import PCA, TruncatedSVD
        if solver == "arpack":
            pca_engine = TruncatedSVD(n_components=self.n_pcs, random_state=42)
        else:
            pca_engine = PCA(n_components=self.n_pcs, svd_solver=solver, random_state=42)
            
        adata.obsm["X_pca_chitin"] = pca_engine.fit_transform(X_centered).astype(np.float32)
        
        # pca_engine.components_ is (n_pcs x n_genes). We transpose it to match your (n_genes x n_pcs) logic.
        self.pca_components = pca_engine.components_.T.astype(np.float32)

        # 4. Save the explicitly centered matrix for V computation in PC decomp
        if self.correction_mode in ("pc_decomposition", "hybrid"):
            X_scaled_for_V = X_centered
        else:
            X_scaled_for_V = None

        force_gc(logger)

        # ── PC decomposition: identify systematic PCs ─────────────────────
        if self.correction_mode in ("pc_decomposition", "hybrid"):
            logger.info("  Identifying systematic PCs via V-alignment (scaled space)...")
            self._fit_pc_decomposition(
                adata, cfg, logger, ctrl_idx, pert_idx,
                X_scaled=X_scaled_for_V)
            del X_scaled_for_V
            gc.collect()

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

        # Fix 2 (Hybrid Contradiction): project systematic PCs out of the
        # control reference matrix so it lives on the same purified manifold
        # as the perturbed residuals during transform().
        # Without this: delta = X_pert_projected - X_ctrl_raw re-injects
        # the systematic variance that was just projected out.
        if (self.correction_mode == "hybrid" and
                len(self.systematic_pc_indices) > 0):
            logger.info("  Hybrid mode: applying PC correction to control "
                        "reference matrix (purifying ctrl manifold)...")
            self.X_ctrl_expr = self._apply_pc_correction(
                self.X_ctrl_expr, logger, "  [ctrl-ref]")

        self.is_fitted = True
        logger.info("  CHITIN v2 model fitted.")
        log_memory(logger, "post fit")
        return adata

    # ─────────────────────────────────────────────────────────────────────
    # STATELESS ALGEBRA HELPERS (For Sweep & Fit)
    # ─────────────────────────────────────────────────────────────────────

    def _project_out_loadings(self, X, loadings):
        """Stateless linear algebra projection for PC decomposition."""
        if len(loadings) == 0:
            return X.copy()
        X_corr = X.copy()
        for l in loadings:
            X_corr = X_corr - np.outer(X_corr @ l, l)
        return X_corr

    def _compute_systematic_pcs(self, X_scaled, ctrl_idx, pert_idx, pca_comps, cfg):
        """Stateless logic to identify systematic PCs for the Pareto sweep."""
        C_ctrl = X_scaled[ctrl_idx].mean(axis=0)
        O_pert = X_scaled[pert_idx].mean(axis=0)
        V = O_pert - C_ctrl
        V_norm = np.linalg.norm(V)

        if V_norm < 1e-10:
            return [], np.zeros((0, X_scaled.shape[1]), dtype=np.float32)

        V_unit = V / V_norm
        n_pcs = pca_comps.shape[1]
        cos_sims = np.zeros(n_pcs)
        for i in range(n_pcs):
            pc = pca_comps[:, i]
            pc_norm = np.linalg.norm(pc)
            if pc_norm > 1e-10:
                cos_sims[i] = abs(float(np.dot(pc / pc_norm, V_unit)))

        threshold = cfg.get("correction", {}).get("pc_systematic_threshold", PC_SYSTEMATIC_COS_THRESHOLD)
        systematic = np.where(cos_sims > threshold)[0]
        if len(systematic) == 0 and cos_sims.max() > 0.2:
            systematic = np.array([np.argmax(cos_sims)])

        sys_idx = systematic.tolist()
        if len(sys_idx) > 0:
            loadings = pca_comps[:, systematic].T.astype(np.float32)
            norms = np.linalg.norm(loadings, axis=1, keepdims=True)
            sys_loadings = loadings / np.where(norms < 1e-10, 1.0, norms)
        else:
            sys_loadings = np.zeros((0, X_scaled.shape[1]), dtype=np.float32)

        return sys_idx, sys_loadings

    # ─────────────────────────────────────────────────────────────────────
    # PC DECOMPOSITION FIT
    # ─────────────────────────────────────────────────────────────────────

    def _fit_pc_decomposition(self, adata, cfg, logger, ctrl_idx, pert_idx,
                              X_scaled=None):
        """
        Identify which PCs encode systematic variation by comparing their
        loading vectors against the systematic variation vector V.

        V = O_pert - C_ctrl  (the average perturbation shift in gene space)

        A PC is classified as systematic if:
          cosine_similarity(PC_loading, V) > PC_SYSTEMATIC_COS_THRESHOLD

        Fix 1 (Space Mismatch): X_scaled must be the same scaled matrix that
        was passed to sc.pp.pca. PC loading vectors are directions in SCALED
        gene space (units of std-normalized expression). Computing V from
        raw adata.X (units of raw counts) gives a vector in a different
        coordinate system, making the cosine similarities meaningless.
        """
        if X_scaled is not None:
            X = X_scaled.astype(np.float32)
        else:
            # Fallback (standalone mode, no pre-scaled matrix available)
            X = adata.X
            if sp.issparse(X):
                X = X.toarray()
            X = np.clip(X.astype(np.float32), None, 10.0)

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

    def _pareto_sweep(self, adata, cfg, logger, k_grid, npcs_grid, metric_grid, ctrl_idx, pert_idx):
        """
        Parallelized multi-mode sweep. If explore mode is active, sweeps across 
        KNN, PC Decomposition, and Hybrid strategies.
        """
        pert_col   = cfg["dataset"]["perturbation_col"]
        ctrl_label = cfg["dataset"]["control_label"]
        solver     = cfg.get("knn", {}).get("svd_solver", "randomized")
        n_jobs     = cfg.get("runtime", {}).get("n_jobs", 8)

        # Subsample for speed
        rng = np.random.default_rng(42)
        n_pert_sub = min(SWEEP_SUBSAMPLE_CELLS, len(pert_idx))
        pert_sub   = rng.choice(pert_idx, size=n_pert_sub, replace=False)
        n_gene_sub = min(SWEEP_SUBSAMPLE_GENES, adata.n_vars)
        gene_sub   = rng.choice(adata.n_vars, size=n_gene_sub, replace=False)

        X_full = adata.X.toarray().astype(np.float32) if sp.issparse(adata.X) else adata.X.astype(np.float32)
        X_ctrl_full = X_full[ctrl_idx]
        X_pert_sub  = X_full[pert_sub]
        X_pert_gene_sub = X_pert_sub[:, gene_sub]

        pre_dists = _fast_pairwise_cosine(X_pert_sub, cfg, pert_col, ctrl_label, adata, pert_sub, SWEEP_N_PAIRS, rng)
        
        # Determine modes to evaluate
        modes_to_test = ['knn', 'pc_decomposition', 'hybrid'] if self.correction_mode == 'explore' else [self.correction_mode]
        
        logger.info(f"  Sweep: Modes={modes_to_test} | {len(k_grid)} k × {len(npcs_grid)} n_pcs × {len(metric_grid)} metrics")
        records = []
        best_pca_cache = {}

        for n_pcs in npcs_grid:
            # ── 1. Cache PCA and Sys_Loadings per n_pcs ──
            if n_pcs not in best_pca_cache:
                sweep_pca_mean = X_full.mean(axis=0)
                X_sweep_centered = X_full - sweep_pca_mean
                
                pca_engine = TruncatedSVD(n_components=n_pcs, random_state=42) if solver == "arpack" else PCA(n_components=n_pcs, svd_solver=solver, random_state=42)
                pca_all = pca_engine.fit_transform(X_sweep_centered).astype(np.float32)
                pca_loadings = pca_engine.components_.T.astype(np.float32)
                
                _, sys_loadings = self._compute_systematic_pcs(X_sweep_centered, ctrl_idx, pert_sub, pca_loadings, cfg)
                best_pca_cache[n_pcs] = (pca_all, sys_loadings)
            else:
                pca_all, sys_loadings = best_pca_cache[n_pcs]

            pca_ctrl_all = pca_all[ctrl_idx]
            pca_pert_sub = pca_all[pert_sub]

            # ── 2. Evaluate specific algorithmic modes ──
            for mode in modes_to_test:
                
                # Mode A: Pure PC Decomposition (Bypasses KNN entirely)
                if mode == 'pc_decomposition':
                    delta = self._project_out_loadings(X_pert_sub, sys_loadings)
                    rec = self._evaluate_delta(delta, X_pert_gene_sub, gene_sub, pre_dists, rng, 0, n_pcs, "N/A", mode)
                    records.append(rec)
                    continue
                
                # Mode B/C: KNN or Hybrid
                X_pert_eval = self._project_out_loadings(X_pert_sub, sys_loadings) if mode == 'hybrid' else X_pert_sub
                X_ctrl_eval = self._project_out_loadings(X_ctrl_full, sys_loadings) if mode == 'hybrid' else X_ctrl_full

                for metric in metric_grid:
                    nn = NearestNeighbors(n_neighbors=max(k_grid), metric=metric, n_jobs=n_jobs)
                    nn.fit(pca_ctrl_all)
                    distances_all, nbr_idx_all = nn.kneighbors(pca_pert_sub)

                    # Joblib Parallel execution over the k-grid using NumPy Vectorization
                    results = Parallel(n_jobs=n_jobs, backend="threading")(
                        delayed(self._sweep_k_worker)(
                            k, n_pcs, metric, mode, X_ctrl_eval, X_pert_eval, 
                            X_pert_gene_sub, gene_sub, nbr_idx_all, pre_dists, rng
                        ) for k in k_grid if k < len(ctrl_idx)
                    )
                    records.extend([r for r in results if r is not None])

        del best_pca_cache
        gc.collect()

        results_df = pd.DataFrame(records)
        if len(results_df) == 0:
            logger.warning("  Sweep produced no valid results, using defaults")
            return results_df, results_df, {"mode": self.correction_mode, "k": cfg["knn"]["k"], "n_pcs": cfg["knn"]["n_pcs"], "metric": cfg["knn"]["distance_metric"]}

        pareto_df = _compute_pareto_front(results_df)
        selected = _select_from_pareto(pareto_df, results_df, cfg, logger)
        return results_df, pareto_df, selected

    def _sweep_k_worker(self, k, n_pcs, metric, mode, X_ctrl_eval, X_pert_eval, X_pert_genes, gene_sub, nbr_idx_all, pre_dists, rng):
        """Worker function for parallelizing the k-grid evaluations using vectorized subtraction."""
        try:
            N_i = X_ctrl_eval[nbr_idx_all[:, :k]].mean(axis=1)
            delta = X_pert_eval - N_i
            return self._evaluate_delta(delta, X_pert_genes, gene_sub, pre_dists, rng, k, n_pcs, metric, mode)
        except Exception:
            return None

    def _evaluate_delta(self, delta, X_pert_genes, gene_sub, pre_dists, rng, k, n_pcs, metric, mode):
        """Unified statistical evaluation logic for all delta matrices, regardless of generation mode."""
        delta_genes = delta[:, gene_sub]
        rhos = []
        for gi in range(len(gene_sub)):
            pre_vals, post_vals = X_pert_genes[:, gi], delta_genes[:, gi]
            if np.std(pre_vals) > 1e-10 and np.std(post_vals) > 1e-10:
                r, _ = spearmanr(pre_vals, post_vals)
                if np.isfinite(r): rhos.append(r)
                
        disruption = 1.0 - float(np.mean(rhos)) if rhos else 1.0
        delta_norms = np.linalg.norm(delta, axis=1)
        signal_stab = float(1.0 / (np.std(delta_norms) + 1e-6))

        n_pairs = min(SWEEP_N_PAIRS, len(pre_dists))
        post_d = []
        if n_pairs > 0:
            sample_pairs = rng.integers(0, len(delta), size=(n_pairs * 2, 2))
            for pa, pb in sample_pairs[:n_pairs]:
                if pa == pb: continue
                ca, cb = delta[pa], delta[pb]
                na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
                if na > 1e-12 and nb > 1e-12:
                    post_d.append(1.0 - float(np.dot(ca / na, cb / nb)))

        pre_mean = float(np.mean(pre_dists)) if len(pre_dists) > 0 else 1.0
        disc_ratio = (float(np.mean(post_d)) if len(post_d) > 0 else pre_mean) / (pre_mean + 1e-10)
        disc_preservation = 1.0 / (1.0 + abs(disc_ratio - 1.0))

        return {
            "mode":               mode,
            "k":                  k,
            "n_pcs":              n_pcs,
            "metric":             metric,
            "rank_disruption":    disruption,
            "disc_ratio":         disc_ratio,
            "disc_preservation":  disc_preservation,
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

        # ── Apply intra-control KNN subtraction (Fix 4: prevent variance collapse)
        # Without this, every control cell gets X_delta = 0.0.
        # Tree-based GRN models (LightGBM) see a point-mass reference class
        # with zero variance → zero information gain → overfitting.
        # Fix: for each control cell, subtract the mean of its k nearest
        # training-control neighbors, preserving natural stochastic variation.
        # ── Apply intra-control KNN subtraction (Preserving Basal Variance) ──
        delta_ctrl = None
        if (len(ctrl_indices) > 0 and self.nn_model is not None and
                self.X_ctrl_expr is not None):
            n_ctrl_local = len(ctrl_indices)
            pca_ctrl_local = adata.obsm["X_pca_chitin"][ctrl_indices]
            
            # Request k+1 neighbors to safely handle both Train and Val/Test splits
            k_for_ctrl = min(self.k + 1, len(self.X_ctrl_expr))
            if k_for_ctrl >= 2:
                distances, ctrl_nbr_idx = self.nn_model.kneighbors(
                    pca_ctrl_local, n_neighbors=k_for_ctrl)
                
                X_ctrl_local = X[ctrl_indices].astype(np.float32)
                
                # FATAL BUG FIX: PC-Correct the local controls before subtraction
                if self.correction_mode in ("pc_decomposition", "hybrid"):
                    X_ctrl_local = self._apply_pc_correction(X_ctrl_local, logger, tag + " [ctrl_local]")
                    
                N_ctrl = np.zeros((n_ctrl_local, n_genes), dtype=np.float32)
                
                # Vectorized Dynamic Masking via NumPy Advanced Indexing
                is_self_match = distances[:, 0:1] < 1e-7
                valid_idx = np.where(is_self_match, ctrl_nbr_idx[:, 1:self.k+1], ctrl_nbr_idx[:, 0:self.k])
                
                # C-Level Vectorized Baseline Subtraction (100x faster than for-loop)
                N_ctrl = self.X_ctrl_expr[valid_idx].mean(axis=1)
                delta_ctrl = X_ctrl_local - N_ctrl

        # ── Build output AnnData ──────────────────────────────────────────
        X_delta = np.zeros((adata.n_obs, n_genes), dtype=np.float32)
        X_delta[pert_indices] = delta_pert
        if delta_ctrl is not None:
            X_delta[ctrl_indices] = delta_ctrl

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

        # C-Level Vectorized Baseline Subtraction
        N_i = self.X_ctrl_expr[nbr_idx].mean(axis=1)
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
        """Project into the training PCA space using the fitted training mean."""
        if self.pca_components is not None and self.pca_mean is not None:
            X = adata.X
            if sp.issparse(X):
                X = X.toarray()
            X = X.astype(np.float32)
            
            # Geometrically align unseen data to the training origin
            X_centered = X - self.pca_mean
            adata.obsm["X_pca_chitin"] = X_centered @ self.pca_components
        else:
            raise RuntimeError("PCA components or training mean missing. Model not fitted properly.")
        
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
      - rank_disruption   (maximise)
      - disc_preservation (maximise — = 1/(1+|disc_ratio-1|), peaks at ratio=1)
      - signal_stability  (maximise)

    Fix 3 (Pareto Misalignment): uses disc_preservation, not raw disc_ratio.
    Maximising disc_preservation drives disc_ratio toward 1.0 (topology
    preserved). Maximising raw disc_ratio favoured k=1 with ratio=49.78
    (manifold fragmentation).
    """
    col = "disc_preservation" if "disc_preservation" in df.columns else "disc_ratio"
    vals = df[["rank_disruption", col, "signal_stability"]].values
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
    Selects the optimal operating point using Weighted Utopian Point Distance.
    Includes a configurable minimum percentile filter to reject highly unbalanced configurations.
    """
    import numpy as np
    import scipy.stats as stats
    
    # 1. Determine the correct Discrimination column name
    disc_col = "disc_preservation" if "disc_preservation" in pareto_df.columns else "disc_ratio"
    
    norm_df = pareto_df.copy()

    # 2. Minimum Percentile Floor (Safety Net)
    min_pct = cfg.get("calibration", {}).get("min_percentile_threshold", 30.0)
    
    if min_pct > 0.0:
        pct_rd = norm_df['rank_disruption'].apply(lambda x: stats.percentileofscore(all_df['rank_disruption'], x))
        pct_dp = norm_df[disc_col].apply(lambda x: stats.percentileofscore(all_df[disc_col], x))
        pct_ss = norm_df['signal_stability'].apply(lambda x: stats.percentileofscore(all_df['signal_stability'], x))
        
        valid_mask = (pct_rd >= min_pct) & (pct_dp >= min_pct) & (pct_ss >= min_pct)
        
        if valid_mask.sum() > 0:
            norm_df = norm_df[valid_mask].copy()
            logger.info(f"  Applied {min_pct}th percentile threshold: {valid_mask.sum()}/{len(pareto_df)} Pareto points survived.")
        else:
            logger.warning(f"  WARNING: No Pareto points met the {min_pct}th percentile floor across all metrics. Falling back to full Pareto front.")
    
    # 3. Min-Max Normalize against the ENTIRE sweep space (0.0 to 1.0)
    for col in ['rank_disruption', disc_col, 'signal_stability']:
        c_min = all_df[col].min()
        c_max = all_df[col].max()
        if c_max > c_min:
            norm_df[f'norm_{col}'] = (norm_df[col] - c_min) / (c_max - c_min)
        else:
            norm_df[f'norm_{col}'] = 1.0  # Fallback if no variance

    # 4. Define the Weights (Prioritize Rank Disruption slightly)
    w_rd = 1.5  # Rank Disruption (Primary Thesis Objective)
    w_dp = 1.0  # Discrimination Preservation
    w_ss = 1.0  # Signal Stability

    # 5. Calculate Weighted Euclidean Distance to the Utopian Point (1, 1, 1)
    norm_df['utopian_dist'] = np.sqrt(
        w_rd * (1.0 - norm_df['norm_rank_disruption'])**2 +
        w_dp * (1.0 - norm_df[f'norm_{disc_col}'])**2 +
        w_ss * (1.0 - norm_df['norm_signal_stability'])**2
    )

    # 6. Select the point closest to Utopia
    best_idx = norm_df['utopian_dist'].idxmin()
    best = norm_df.loc[best_idx]

    logger.info(f"  Weighted Utopian Point Selection: k={int(best['k'])}, "
                f"n_pcs={int(best['n_pcs'])}, metric={best['metric']}")
    logger.info(f"    rank_disruption={best['rank_disruption']:.4f}  "
                f"disc_ratio={best['disc_ratio']:.4f}  "
                f"stability={best['signal_stability']:.4f}")

    mode_str = best.get('mode', 'knn').upper()
    logger.info(f"  Weighted Utopian Point Selection: MODE={mode_str}")

    return {
        "mode":   best.get("mode", "knn"),
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
