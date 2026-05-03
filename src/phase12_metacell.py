"""
SPORE+ · src/phase12_metacell.py
─────────────────────────────────
Phase 12: Meta-Cell Aggregation & Systema Calibration

KEY CHANGE vs original SPORE phase8:
  MULTI-CELL-LINE OUTER LOOP — when 'sporeplus_cell_line' is present in .obs
  (added by Phase 11), metacells are aggregated PER CELL LINE and output as:
    {dataset}_{cell_line}_{split}_metacell.h5ad

  For single-cell-line datasets the behavior is identical to original SPORE.

QUALITY METRICS (new):
  Inner variance per perturbation group is computed and reported.
  Groups with high inner variance are flagged in the Phase 13 summary.

MEMORY SAFETY:
  All lessons from SPORE error log errors 036-037 are preserved:
  - Empty cluster guard: if c_mask.sum() == 0: continue
  - Micro-copy safety bypass: X_sub = X_sub.copy() for joblib read-only lock
  - Shard checkpointing preserved
"""

import os
import gc
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, Any
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed

from .utils import (log_phase_header, snapshot, log_memory, force_gc,
                    get_cell_line_color)


# ═══════════════════════════════════════════════════════════════════════════
#  PERTURBATION WORKER (parallelized, per cell-line per-perturbation)
# ═══════════════════════════════════════════════════════════════════════════

def _process_perturbation(pert, X_sub, emb_sub, target_k):
    """
    Isolated worker: aggregates single cells from one perturbation group
    into metacells via MiniBatchKMeans on the PCA embedding.

    CRITICAL (Error 037): joblib passes arrays as read-only memory maps.
    Scikit-learn's Cython KMeans REQUIRES writable buffers.
    The .copy() calls below convert the read-only map to writable local RAM.
    These are microscopic slices — the copy overhead is trivial.
    """
    X_sub   = X_sub.copy()
    emb_sub = emb_sub.copy()

    n_cells     = X_sub.shape[0]
    n_metacells = max(1, n_cells // target_k)

    if n_metacells > 1:
        km     = MiniBatchKMeans(
            n_clusters=n_metacells, random_state=42, n_init="auto")
        labels = km.fit_predict(emb_sub)
    else:
        labels = np.zeros(n_cells, dtype=int)

    mc_exprs     = []
    mc_obs_data  = []

    for c in range(n_metacells):
        c_mask = labels == c
        # CRITICAL (Error 036): Empty cluster guard — MiniBatchKMeans can
        # produce empty clusters. Skip them rather than computing mean of 0 rows.
        if c_mask.sum() == 0:
            continue

        if sp.issparse(X_sub):
            expr = np.asarray(X_sub[c_mask].mean(axis=0)).flatten()
        else:
            expr = X_sub[c_mask].mean(axis=0)

        mc_exprs.append(expr)
        mc_obs_data.append({"n_cells_in_metacell": int(c_mask.sum())})

    return pert, mc_exprs, mc_obs_data


# ═══════════════════════════════════════════════════════════════════════════
#  INNER VARIANCE QUALITY METRIC (new in SPORE+)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_inner_variance(pert_exprs, pert_label):
    """
    Normalized inner variance for a set of metacell expression vectors.
    High inner variance = metacells are heterogeneous = aggregation quality is poor.
    Based on Metacell-2 paper quality metric.

    Returns: (mean_normalized_inner_variance, n_metacells)
    """
    if len(pert_exprs) < 2:
        return 0.0, len(pert_exprs)
    mat = np.array(pert_exprs)   # (n_metacells, n_genes)
    gene_means = mat.mean(axis=0)
    gene_means = np.where(gene_means > 1e-10, gene_means, 1.0)
    normalized_var = np.var(mat, axis=0) / gene_means
    return float(normalized_var.mean()), len(pert_exprs)


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEMA CALIBRATION (unchanged from SPORE)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_systema_centroids(adata_meta, cfg: dict, logger):
    pert_col   = cfg.get("dataset", {}).get("perturbation_col", "gene")
    ctrl_label = cfg.get("dataset", {}).get("control_label", "non-targeting")

    ctrl_mask = (adata_meta.obs[pert_col] == ctrl_label).values
    if ctrl_mask.sum() == 0:
        logger.warning("  Systema calibration: no control metacells found")
        return None, None

    C_ctrl        = adata_meta.X[ctrl_mask].mean(axis=0)
    unique_perts  = set(adata_meta.obs[pert_col].unique()) - {ctrl_label}
    pert_centroids = []

    for pert in unique_perts:
        p_mask = (adata_meta.obs[pert_col] == pert).values
        if p_mask.sum() > 0:
            pert_centroids.append(adata_meta.X[p_mask].mean(axis=0))

    if not pert_centroids:
        return C_ctrl, None

    O_pert = np.vstack(pert_centroids).mean(axis=0)
    logger.info("  Systema calibration: C_ctrl and O_pert vectors established")
    return C_ctrl, O_pert


# ═══════════════════════════════════════════════════════════════════════════
#  AGGREGATE ONE SPLIT FOR ONE CELL LINE
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_split(adata, cfg: dict, logger, label: str,
                    cell_line_name: str = ""):
    """
    Aggregate single cells → metacells for one (cell_line, split) combination.

    Uses sharded checkpointing for resumability.
    Computes inner variance quality metrics if enabled.
    """
    p12         = cfg.get("phase12_metacell", {})
    n_jobs      = cfg.get("runtime", {}).get("n_jobs", 8)
    target_k    = p12.get("target_cells_per_metacell", 10)
    pert_col    = cfg.get("dataset", {}).get("perturbation_col", "gene")
    compute_qc  = p12.get("compute_quality_metrics", True)
    var_warn    = p12.get("inner_variance_warn_threshold", 2.0)

    dataset_name = cfg.get("dataset", {}).get("name", "dataset")
    cl_tag       = f"_{cell_line_name}" if cell_line_name else ""
    split_tag    = label.lower()

    # ── Checkpoint setup ───────────────────────────────────────────────────
    checkpoint_dir = cfg["paths"]["_processed"] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shard_prefix = str(
        checkpoint_dir / f"{dataset_name}{cl_tag}_shard_{split_tag}")

    logger.info(
        f"  [{label}{cl_tag}] Aggregating {adata.n_obs:,} cells → metacells "
        f"(target {target_k} cells/mc)")

    # ── Select embedding (Harmony PCA preferred) ───────────────────────────
    if "X_pca_harmony" in adata.obsm:
        use_rep = "X_pca_harmony"
    elif "X_pca" in adata.obsm:
        use_rep = "X_pca"
    else:
        import scanpy as sc
        logger.info(f"  [{label}{cl_tag}] No PCA found. Computing temporary PCA...")
        sc.pp.pca(adata, n_comps=50, use_highly_variable=False,
                  zero_center=False, svd_solver="randomized")
        use_rep = "X_pca"

    logger.info(f"  [{label}{cl_tag}] Clustering via '{use_rep}' embeddings")

    unique_perts   = adata.obs[pert_col].unique()
    new_X_list     = []
    new_obs_list   = []
    quality_records = []

    # ── Sharded execution ──────────────────────────────────────────────────
    chunk_size = 500
    for i in range(0, len(unique_perts), chunk_size):
        pert_chunk = unique_perts[i:i + chunk_size]
        shard_file = f"{shard_prefix}_{i}.npz"

        if os.path.exists(shard_file):
            logger.info(
                f"  [{label}{cl_tag}] Recovering shard {i}–{i+len(pert_chunk)}")
            loaded    = np.load(shard_file, allow_pickle=True)
            new_X_list.extend(loaded["X"])
            for p_val, n_cells in zip(loaded["perts"], loaded["counts"]):
                new_obs_list.append({
                    pert_col: p_val,
                    "n_cells_in_metacell": int(n_cells)})
            continue

        tasks = []
        for pert in pert_chunk:
            pert_mask = (adata.obs[pert_col] == pert).values
            X_sub     = adata.X[pert_mask]
            emb_sub   = adata.obsm[use_rep][pert_mask]
            tasks.append((pert, X_sub, emb_sub, target_k))

        # backend="threading" is CRITICAL (Error 002):
        # loky backend pickles the full matrix to each worker = OOM
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_process_perturbation)(*task) for task in tasks)

        shard_X, shard_perts, shard_counts = [], [], []
        for pert, exprs, obs_data in results:
            if not exprs:
                continue
            shard_X.extend(exprs)
            # Quality metric
            if compute_qc:
                var_val, n_mc = _compute_inner_variance(exprs, pert)
                quality_records.append({
                    "perturbation":          pert,
                    "n_metacells":           n_mc,
                    "inner_variance":        var_val,
                    "high_variance_warning": var_val > var_warn,
                })
            for o in obs_data:
                shard_perts.append(pert)
                shard_counts.append(o["n_cells_in_metacell"])

        np.savez(shard_file,
                 X=np.array(shard_X),
                 perts=np.array(shard_perts),
                 counts=np.array(shard_counts))

        new_X_list.extend(shard_X)
        for p_val, n_cells in zip(shard_perts, shard_counts):
            new_obs_list.append({
                pert_col: p_val,
                "n_cells_in_metacell": int(n_cells)})

        force_gc(logger)

    X_meta   = np.vstack(new_X_list)
    obs_meta = pd.DataFrame(new_obs_list)

    adata_meta = ad.AnnData(X=X_meta, obs=obs_meta, var=adata.var.copy())
    if hasattr(adata, "uns"):
        adata_meta.uns = adata.uns.copy()

    # Store quality metrics
    if quality_records:
        qdf = pd.DataFrame(quality_records)
        # h5py cannot serialize numpy scalars (np.float64, np.bool_, np.int64).
        # Convert to native Python types before storing in .uns so write_h5ad works.
        def _to_python(v):
            import numpy as np
            if isinstance(v, (np.integer,)):  return int(v)
            if isinstance(v, (np.floating,)): return float(v)
            if isinstance(v, (np.bool_,)):    return bool(v)
            return v
        adata_meta.uns["metacell_quality"] = [
            {k: _to_python(v) for k, v in row.items()}
            for row in qdf.to_dict("records")]
        n_flagged = qdf["high_variance_warning"].sum()
        if n_flagged > 0:
            logger.warning(
                f"  [{label}{cl_tag}] ⚠ {n_flagged} perturbation group(s) "
                f"have inner variance > {var_warn}. "
                f"Metacell quality may be insufficient for these groups.")
        logger.info(
            f"  [{label}{cl_tag}] Inner variance: "
            f"mean={qdf['inner_variance'].mean():.4f}, "
            f"{n_flagged} groups flagged")

    logger.info(
        f"  [{label}{cl_tag}] Compressed {adata.n_obs:,} cells → "
        f"{adata_meta.n_obs:,} metacells")
    return adata_meta


# ═══════════════════════════════════════════════════════════════════════════
#  RUN PHASE 12  (multi-cell-line aware)
# ═══════════════════════════════════════════════════════════════════════════

def run_phase12(splits: dict, cfg: dict, logger):
    """
    Aggregate all splits into metacells.

    If 'sporeplus_cell_line' is in splits['train'].obs:
      → Produces {dataset}_{cell_line}_{split}_metacell.h5ad for each combination
      → Returns: dict of { cell_line_name: {train/val/test: adata_meta} }

    If single cell line (or cell line detection not run):
      → Produces {dataset}_{split}_metacell.h5ad
      → Returns: dict of { "single": {train/val/test: adata_meta} }
    """
    log_phase_header(logger, 12, "Meta-Cell Aggregation & Systema Calibration")
    p12          = cfg.get("phase12_metacell", {})
    dataset_name = cfg.get("dataset", {}).get("name", "dataset")
    out_dir      = cfg["paths"]["_processed"]
    suffix       = p12.get("suffix", "_metacell")

    if not p12.get("enabled", True):
        logger.info("  Phase 12: DISABLED in config")
        return {"single": splits}

    train_split   = splits.get("train")
    has_cell_line = (
        train_split is not None and
        "sporeplus_cell_line" in train_split.obs.columns)

    if has_cell_line:
        unique_lines = sorted(
            train_split.obs["sporeplus_cell_line"].unique().tolist())
        min_cells = cfg.get("phase11_cell_line", {}).get("min_cells_per_cell_line", 500)
        # Filter out small cell lines
        excluded = cfg.get("_detection", {}).get("excluded_small_lines", [])
        unique_lines = [l for l in unique_lines if l not in excluded]
        logger.info(
            f"  Multi-cell-line mode: {len(unique_lines)} cell line(s): "
            f"{unique_lines[:8]}")
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

            # Subset to this cell line only
            if has_cell_line and cell_line != "single":
                cl_mask = split_adata.obs["sporeplus_cell_line"] == cell_line
                n_cells = cl_mask.sum()
                if n_cells < 50:
                    logger.warning(
                        f"  Skipping {cell_line}/{split_key}: only {n_cells} cells")
                    continue
                # Use safe_subset to avoid 2x RAM spike
                from .utils import safe_subset
                split_subset = safe_subset(
                    split_adata, cell_mask=cl_mask, logger=logger)
            else:
                split_subset = split_adata

            adata_meta = aggregate_split(
                split_subset, cfg, logger,
                label=split_key.title(),
                cell_line_name=cell_line if cell_line != "single" else "")
            line_splits[split_key] = adata_meta

        # ── Systema calibration on train split ─────────────────────────────
        if p12.get("systema_calibration", True) and "train" in line_splits:
            C_ctrl, O_pert = calculate_systema_centroids(
                line_splits["train"], cfg, logger)
            if C_ctrl is not None and O_pert is not None:
                for sp_key, sp_adata in line_splits.items():
                    sp_adata.uns["systema_C_ctrl"] = C_ctrl
                    sp_adata.uns["systema_O_pert"] = O_pert

        # ── Save to disk ────────────────────────────────────────────────────
        cl_tag = f"_{cell_line}" if cell_line != "single" else ""
        for split_key, adata_meta in line_splits.items():
            fname = f"{dataset_name}{cl_tag}_{split_key}{suffix}.h5ad"
            path  = out_dir / fname
            adata_meta.write_h5ad(path)
            snapshot(adata_meta, f"Saved {cell_line}/{split_key}", logger)
            logger.info(f"  ✓ Saved → {path}")

        all_meta_splits[cell_line] = line_splits
        force_gc(logger)

    return all_meta_splits

# ─────────────────────────────────────────────────────────────────────────────
#  Bug 1 fix: uns sanitizer
# ─────────────────────────────────────────────────────────────────────────────
 
def _sanitize_uns(uns: dict) -> dict:
    """
    Deep-clean adata.uns so it can be serialised by anndata's write_h5ad.
    Converts tuple/int/float keys to strings, converts numpy scalars and
    non-serialisable leaf values to Python native types.
    """
    import numpy as np
 
    def _make_key(k):
        if isinstance(k, str):
            return k
        return str(k)
 
    def _clean_value(v):
        if isinstance(v, dict):
            return {_make_key(kk): _clean_value(vv) for kk, vv in v.items()}
        if isinstance(v, (list, tuple)):
            cleaned = [_clean_value(i) for i in v]
            return cleaned
        if isinstance(v, np.ndarray):
            return v   # arrays are fine for anndata
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
 
    return {_make_key(k): _clean_value(v) for k, v in uns.items()}
 
 
def patch_anndata_write():
    """
    Monkey-patch anndata.AnnData.write_h5ad to sanitise uns before writing.
    Call once before run_phase12.
    This is safe to call multiple times (idempotent guard included).
    """
    import anndata as ad
 
    if getattr(ad.AnnData, "_sporeplus_patched", False):
        return
 
    _original = ad.AnnData.write_h5ad
 
    def _safe_write_h5ad(self, *args, **kwargs):
        self.uns = _sanitize_uns(self.uns)
        return _original(self, *args, **kwargs)
 
    ad.AnnData.write_h5ad = _safe_write_h5ad
    ad.AnnData._sporeplus_patched = True
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Bug 2 fix: propagate cell line labels to reloaded splits
# ─────────────────────────────────────────────────────────────────────────────
 
def apply_cell_line_labels_to_splits(splits: dict, cell_line_meta: dict,
                                      cfg: dict, logger) -> None:
    """
    Copy the cell_type_col from each reloaded split into 'sporeplus_cell_line'
    so that run_phase12 correctly detects multi-cell-line mode.
 
    Mutates splits in-place.
 
    Parameters
    ----------
    splits       : dict with keys 'train', 'val', 'test'
    cell_line_meta : dict returned by run_phase11
    cfg          : SPORE+ config
    logger       : logger instance
    """
    n_lines  = cell_line_meta.get("n_cell_lines", 1)
    cl_col   = cell_line_meta.get("cell_type_col")
 
    if n_lines <= 1 or not cl_col:
        logger.info("  Phase 12 prep: single cell-line mode (no label propagation needed)")
        return
 
    logger.info(f"  Phase 12 prep: propagating '{cl_col}' → 'sporeplus_cell_line' "
                f"for {n_lines} cell lines across all splits")
 
    for key in ["train", "val", "test"]:
        if key not in splits:
            continue
        split = splits[key]
        if cl_col not in split.obs.columns:
            logger.warning(f"  Phase 12 prep: '{cl_col}' not in {key}.obs — "
                           "cell line separation may not work correctly")
            continue
        split.obs["sporeplus_cell_line"] = split.obs[cl_col].astype(str)
        n_labeled = split.obs["sporeplus_cell_line"].nunique()
        logger.info(f"    {key}: {n_labeled} unique cell lines labeled")
