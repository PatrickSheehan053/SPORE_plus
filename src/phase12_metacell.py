"""
SPORE+ · src/phase12_metacell.py
─────────────────────────────────
Phase 12: Meta-Cell Aggregation & Systema Calibration

MILESTONE 2 UPGRADES:
  Dynamic graining: target_cells_per_metacell scales with group size.
    < 30 cells  → 1 mc (no clustering)
    30-100      → 5 cells/mc
    100-500     → 10 cells/mc (default)
    > 500       → 15 cells/mc
  This prevents single-cell metacells for tiny groups and wasteful over-splitting
  of large groups.

  Per-group mini-PCA: for groups > 50 cells, compute a local 10-PC PCA on just
  that group's cells for MiniBatchKMeans clustering. This captures the local
  transcriptional structure within the perturbation better than the global
  Harmony PCA (which captures cell line and batch effects more than
  perturbation-specific variation).

  uns sanitizer: all .uns values are deep-cleaned before write_h5ad to prevent
  h5py TypeError on numpy scalars.

  apply_cell_line_labels_to_splits: fixed key chain (source_col > cell_type_col).
"""

import os
import gc
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed

from .utils import (log_phase_header, snapshot, log_memory, force_gc,
                    get_cell_line_color, safe_subset)


# ═══════════════════════════════════════════════════════════════════════════
#  DYNAMIC GRAINING
# ═══════════════════════════════════════════════════════════════════════════

def _dynamic_target_k(n_cells: int, base_target: int = 10) -> int:
    """
    Scale cells-per-metacell based on group size.
    Prevents both 1-cell metacells (too few cells) and excessive splitting.
    """
    if n_cells < 30:
        return n_cells   # no clustering — one metacell per cell
    if n_cells < 100:
        return 5
    if n_cells < 500:
        return base_target
    return min(base_target + 5, 20)   # max 20 cells/mc for large groups


# ═══════════════════════════════════════════════════════════════════════════
#  PER-GROUP MINI-PCA
# ═══════════════════════════════════════════════════════════════════════════

def _compute_group_pca(X_sub, n_comps: int = 10) -> np.ndarray:
    """
    Compute a mini PCA on a single perturbation group's expression matrix.
    Falls back to returning None if X_sub is too small or TruncatedSVD fails.
    Returns (n_cells, n_comps) float32 embedding or None.
    """
    from sklearn.decomposition import TruncatedSVD
    n_cells, n_genes = X_sub.shape
    n_comps_actual   = min(n_comps, n_cells - 1, n_genes - 1)
    if n_comps_actual < 2:
        return None
    try:
        svd = TruncatedSVD(n_components=n_comps_actual,
                           algorithm="randomized", n_iter=2, random_state=42)
        if sp.issparse(X_sub):
            return svd.fit_transform(X_sub).astype(np.float32)
        else:
            return svd.fit_transform(X_sub).astype(np.float32)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  PERTURBATION WORKER
# ═══════════════════════════════════════════════════════════════════════════

_PER_GROUP_PCA_THRESHOLD = 50   # groups with >= this many cells get mini-PCA

def _process_perturbation(pert, X_sub, emb_sub, base_target_k):
    """
    Aggregate one perturbation group into metacells.

    M2 upgrades:
      - Dynamic graining: target_k scales with n_cells
      - Per-group PCA: if n_cells >= threshold, compute local PCA for clustering
      - Error 037: .copy() on read-only memory maps from joblib
    """
    X_sub   = X_sub.copy()
    n_cells = X_sub.shape[0]

    # Dynamic graining
    target_k    = _dynamic_target_k(n_cells, base_target_k)
    n_metacells = max(1, n_cells // target_k)

    if n_metacells > 1:
        # Per-group PCA for clustering (M2)
        if n_cells >= _PER_GROUP_PCA_THRESHOLD:
            local_emb = _compute_group_pca(X_sub, n_comps=10)
            if local_emb is not None:
                cluster_emb = local_emb
            else:
                cluster_emb = emb_sub.copy()
        else:
            cluster_emb = emb_sub.copy()

        km     = MiniBatchKMeans(n_clusters=n_metacells,
                                  random_state=42, n_init="auto")
        labels = km.fit_predict(cluster_emb)
    else:
        labels = np.zeros(n_cells, dtype=int)

    mc_exprs    = []
    mc_obs_data = []
    for c in range(n_metacells):
        c_mask = labels == c
        if c_mask.sum() == 0:   # Error 036: empty cluster guard
            continue
        if sp.issparse(X_sub):
            expr = np.asarray(X_sub[c_mask].mean(axis=0)).flatten()
        else:
            expr = X_sub[c_mask].mean(axis=0)
        mc_exprs.append(expr)
        mc_obs_data.append({"n_cells_in_metacell": int(c_mask.sum())})

    return pert, mc_exprs, mc_obs_data


# ═══════════════════════════════════════════════════════════════════════════
#  INNER VARIANCE QUALITY METRIC
# ═══════════════════════════════════════════════════════════════════════════

def _compute_inner_variance(pert_exprs):
    if len(pert_exprs) < 2:
        return 0.0, len(pert_exprs)
    mat        = np.array(pert_exprs)
    gene_means = mat.mean(axis=0)
    gene_means = np.where(gene_means > 1e-10, gene_means, 1.0)
    return float(np.var(mat, axis=0).mean() / gene_means.mean()), len(pert_exprs)


# ═══════════════════════════════════════════════════════════════════════════
#  UNS SANITIZER (prevents h5py TypeError on numpy scalars)
# ═══════════════════════════════════════════════════════════════════════════

def _sanitize_uns(uns: dict) -> dict:
    def _key(k):
        return k if isinstance(k, str) else str(k)
    def _val(v):
        if isinstance(v, dict):
            return {_key(kk): _val(vv) for kk, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [_val(i) for i in v]
        if isinstance(v, np.ndarray):
            return v
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, np.floating): return float(v)
        if isinstance(v, np.bool_): return bool(v)
        return v
    return {_key(k): _val(v) for k, v in uns.items()}


def patch_anndata_write():
    """Monkey-patch AnnData.write_h5ad to sanitise uns. Idempotent."""
    import anndata as _ad
    if getattr(_ad.AnnData, "_sporeplus_patched", False):
        return
    _orig = _ad.AnnData.write_h5ad
    def _safe(self, *args, **kwargs):
        self.uns = _sanitize_uns(self.uns)
        return _orig(self, *args, **kwargs)
    _ad.AnnData.write_h5ad      = _safe
    _ad.AnnData._sporeplus_patched = True


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEMA CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

def calculate_systema_centroids(adata_meta, cfg: dict, logger):
    pert_col   = cfg.get("dataset", {}).get("perturbation_col", "gene")
    ctrl_label = cfg.get("dataset", {}).get("control_label", "non-targeting")
    ctrl_mask  = (adata_meta.obs[pert_col] == ctrl_label).values
    if ctrl_mask.sum() == 0:
        logger.warning("  Systema calibration: no control metacells")
        return None, None
    C_ctrl = adata_meta.X[ctrl_mask].mean(axis=0)
    perts  = [p for p in adata_meta.obs[pert_col].unique() if p != ctrl_label]
    cents  = [adata_meta.X[(adata_meta.obs[pert_col] == p).values].mean(axis=0)
              for p in perts if (adata_meta.obs[pert_col] == p).sum() > 0]
    if not cents:
        return C_ctrl, None
    O_pert = np.vstack(cents).mean(axis=0)
    logger.info("  Systema calibration: C_ctrl and O_pert established")
    return C_ctrl, O_pert


# ═══════════════════════════════════════════════════════════════════════════
#  AGGREGATE ONE SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_split(adata, cfg: dict, logger, label: str, cell_line_name: str = ""):
    p12         = cfg.get("phase12_metacell", {})
    n_jobs      = cfg.get("runtime", {}).get("n_jobs", 8)
    base_target = p12.get("target_cells_per_metacell", 10)
    pert_col    = cfg.get("dataset", {}).get("perturbation_col", "gene")
    compute_qc  = p12.get("compute_quality_metrics", True)
    var_warn    = p12.get("inner_variance_warn_threshold", 2.0)
    dataset_name = cfg.get("dataset", {}).get("name", "dataset")
    cl_tag       = f"_{cell_line_name}" if cell_line_name else ""

    checkpoint_dir = cfg["paths"]["_processed"] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shard_prefix = str(checkpoint_dir / f"{dataset_name}{cl_tag}_shard_{label.lower()}")

    logger.info(f"  [{label}{cl_tag}] {adata.n_obs:,} cells → metacells "
                f"(dynamic graining, base={base_target} cells/mc)")

    # Embedding selection
    if "X_pca_harmony" in adata.obsm:
        use_rep = "X_pca_harmony"
    elif "X_pca" in adata.obsm:
        use_rep = "X_pca"
    else:
        import scanpy as sc
        logger.info(f"  [{label}{cl_tag}] No PCA — computing temporary PCA...")
        sc.pp.pca(adata, n_comps=50, use_highly_variable=False,
                  zero_center=False, svd_solver="randomized")
        use_rep = "X_pca"

    logger.info(f"  [{label}{cl_tag}] Global embedding: {use_rep} "
                f"(per-group PCA for groups ≥ {_PER_GROUP_PCA_THRESHOLD} cells)")

    unique_perts    = adata.obs[pert_col].unique()
    new_X_list      = []
    new_obs_list    = []
    quality_records = []

    chunk_size = 500
    for i in range(0, len(unique_perts), chunk_size):
        pert_chunk = unique_perts[i:i + chunk_size]
        shard_file = f"{shard_prefix}_{i}.npz"

        if os.path.exists(shard_file):
            logger.info(f"  [{label}{cl_tag}] Recovering shard {i}–{i+len(pert_chunk)}")
            loaded = np.load(shard_file, allow_pickle=True)
            new_X_list.extend(loaded["X"])
            for p_val, n_c in zip(loaded["perts"], loaded["counts"]):
                new_obs_list.append({pert_col: p_val, "n_cells_in_metacell": int(n_c)})
            continue

        tasks = []
        for pert in pert_chunk:
            pert_mask = (adata.obs[pert_col] == pert).values
            tasks.append((pert, adata.X[pert_mask], adata.obsm[use_rep][pert_mask], base_target))

        # backend="threading" is critical (Error 002: loky pickles full matrix → OOM)
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_process_perturbation)(*t) for t in tasks)

        shard_X, shard_perts, shard_counts = [], [], []
        for pert, exprs, obs_data in results:
            if not exprs:
                continue
            shard_X.extend(exprs)
            if compute_qc:
                iv, n_mc = _compute_inner_variance(exprs)
                quality_records.append({
                    "perturbation": pert, "n_metacells": n_mc,
                    "inner_variance": iv, "high_variance_warning": iv > var_warn})
            for o in obs_data:
                shard_perts.append(pert)
                shard_counts.append(o["n_cells_in_metacell"])

        np.savez(shard_file, X=np.array(shard_X),
                 perts=np.array(shard_perts), counts=np.array(shard_counts))
        new_X_list.extend(shard_X)
        for p_val, n_c in zip(shard_perts, shard_counts):
            new_obs_list.append({pert_col: p_val, "n_cells_in_metacell": int(n_c)})
        force_gc(logger)

    X_meta   = np.vstack(new_X_list)
    obs_meta = pd.DataFrame(new_obs_list)
    adata_meta = ad.AnnData(X=X_meta, obs=obs_meta, var=adata.var.copy())
    if hasattr(adata, "uns"):
        adata_meta.uns = _sanitize_uns(adata.uns.copy())

    if quality_records:
        qdf      = pd.DataFrame(quality_records)
        n_flagged = int(qdf["high_variance_warning"].sum())
        adata_meta.uns["metacell_quality"] = [
            {k: (int(v) if isinstance(v, (np.integer,)) else
                 float(v) if isinstance(v, (np.floating,)) else
                 bool(v) if isinstance(v, (np.bool_,)) else v)
             for k, v in row.items()}
            for row in qdf.to_dict("records")]
        if n_flagged > 0:
            logger.warning(f"  [{label}{cl_tag}] ⚠ {n_flagged} groups inner_variance > {var_warn}")
        logger.info(f"  [{label}{cl_tag}] Inner variance: mean={qdf['inner_variance'].mean():.4f}, "
                    f"{n_flagged} flagged")

    logger.info(f"  [{label}{cl_tag}] {adata.n_obs:,} → {adata_meta.n_obs:,} metacells")
    return adata_meta


# ═══════════════════════════════════════════════════════════════════════════
#  APPLY CELL LINE LABELS  (fixed key chain)
# ═══════════════════════════════════════════════════════════════════════════

def apply_cell_line_labels_to_splits(splits: dict, cell_line_meta: dict,
                                      cfg: dict, logger) -> None:
    """
    Propagate cell line labels to reloaded p9 splits so Phase 12 runs
    in multi-cell-line mode. Mutates splits in-place.

    Key fix: Phase 11 stores the source column as "source_col" (Tier 1),
    not "cell_type_col" as the other agent assumed.
    """
    n_lines = cell_line_meta.get("n_cell_lines", 1)
    # Fixed: try all possible key names that Phase 11 might use
    cl_col  = (cell_line_meta.get("source_col") or
               cell_line_meta.get("cell_type_col") or
               cell_line_meta.get("detected_cell_line_col"))

    if n_lines <= 1 or not cl_col:
        logger.info("  Phase 12 prep: single cell-line mode")
        return

    logger.info(f"  Phase 12 prep: propagating '{cl_col}' → 'sporeplus_cell_line' "
                f"({n_lines} cell lines)")

    for key in ["train", "val", "test"]:
        if key not in splits:
            continue
        split = splits[key]
        if cl_col in split.obs.columns:
            split.obs["sporeplus_cell_line"] = split.obs[cl_col].astype(str)
            n_u = split.obs["sporeplus_cell_line"].nunique()
            logger.info(f"    {key}: {n_u} cell lines labeled from '{cl_col}'")
        else:
            logger.warning(f"  ⚠ '{cl_col}' not in {key}.obs — "
                           "sporeplus_cell_line not set for this split")


# ═══════════════════════════════════════════════════════════════════════════
#  RUN PHASE 12
# ═══════════════════════════════════════════════════════════════════════════

def run_phase12(splits: dict, cfg: dict, logger):
    log_phase_header(logger, 12, "Meta-Cell Aggregation & Systema Calibration")
    p12          = cfg.get("phase12_metacell", {})
    dataset_name = cfg.get("dataset", {}).get("name", "dataset")
    out_dir      = cfg["paths"]["_processed"]
    suffix       = p12.get("suffix", "_metacell")

    if not p12.get("enabled", True):
        logger.info("  Phase 12: DISABLED")
        return {"single": splits}

    train_split   = splits.get("train")
    has_cell_line = (train_split is not None and
                     "sporeplus_cell_line" in train_split.obs.columns)

    if has_cell_line:
        excluded     = cfg.get("_detection", {}).get("excluded_small_lines", [])
        unique_lines = [l for l in sorted(
            train_split.obs["sporeplus_cell_line"].unique().tolist())
            if l not in excluded]
        logger.info(f"  Multi-cell-line: {len(unique_lines)} lines: {unique_lines[:8]}")
    else:
        unique_lines = ["single"]
        logger.info("  Single cell-line mode")

    all_meta_splits = {}

    for cell_line in unique_lines:
        logger.info(f"  ── Cell line: {cell_line} ──")
        line_splits = {}

        for split_key in ["train", "val", "test"]:
            split_adata = splits.get(split_key)
            if split_adata is None:
                continue
            if has_cell_line and cell_line != "single":
                cl_mask = split_adata.obs["sporeplus_cell_line"] == cell_line
                n_cells = cl_mask.sum()
                if n_cells < 50:
                    logger.warning(f"  Skipping {cell_line}/{split_key}: {n_cells} cells")
                    continue
                split_subset = safe_subset(split_adata, cell_mask=cl_mask, logger=logger)
            else:
                split_subset = split_adata

            adata_meta = aggregate_split(
                split_subset, cfg, logger,
                label=split_key.title(),
                cell_line_name=cell_line if cell_line != "single" else "")
            line_splits[split_key] = adata_meta

        if p12.get("systema_calibration", True) and "train" in line_splits:
            C_ctrl, O_pert = calculate_systema_centroids(line_splits["train"], cfg, logger)
            if C_ctrl is not None and O_pert is not None:
                for sp_adata in line_splits.values():
                    sp_adata.uns["systema_C_ctrl"] = (C_ctrl.tolist() if hasattr(C_ctrl, "tolist") else C_ctrl)
                    sp_adata.uns["systema_O_pert"] = (O_pert.tolist() if hasattr(O_pert, "tolist") else O_pert)

        cl_tag = f"_{cell_line}" if cell_line != "single" else ""
        for split_key, adata_meta in line_splits.items():
            fname = f"{dataset_name}{cl_tag}_{split_key}{suffix}.h5ad"
            path  = out_dir / fname
            adata_meta.uns = _sanitize_uns(adata_meta.uns)
            adata_meta.write_h5ad(path)
            snapshot(adata_meta, f"Saved {cell_line}/{split_key}", logger)
            logger.info(f"  ✓ Saved → {path}")

        all_meta_splits[cell_line] = line_splits
        force_gc(logger)

    return all_meta_splits
