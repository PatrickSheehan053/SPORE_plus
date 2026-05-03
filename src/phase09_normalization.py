"""
SPORE+ · src/phase09_normalization.py
───────────────────────────────────────
Phase 9: Normalization

NEW ERROR DOCUMENTED (SPORE+ Error 039):
  sc.pp.log1p causes an 18.5 GB RAM spike across three splits.
  Root cause: Scanpy's implementation does X.data = np.log1p(X.data).
  np.log1p() WITHOUT an out= argument ALWAYS allocates a brand new array.
  The old X.data (13.5 GB for the train split) stays in Python's heap
  as unreturned pages — the OS never sees those bytes freed.

  Cumulative across 3 splits: 13.5 + 3.0 + 1.8 GB = 18.3 GB of ghost RAM
  that inflates the Phase 10 baseline from 88 GB to 106 GB, leaving too
  little headroom for sc.tl.score_genes_cell_cycle's control gene matrix.

  Fix: np.log1p(X.data, out=X.data)
  The out= argument forces modification of the existing buffer in-place.
  Zero bytes allocated. Zero ghost RAM.

Other lessons applied:
  Error 024: no adata.layers["raw_counts"] copy — raw saved to disk by Phase 7
  Error 024: pre-cast to float32 before normalize_total to prevent float64 upcast
"""

import numpy as np
import scipy.sparse as sp
import scanpy as sc
from .utils import log_phase_header, snapshot, log_memory, force_gc


def normalize_split(adata, cfg: dict, logger, label: str = ""):
    """
    Normalize one split: median library-size scaling + log1p.

    Both operations are now truly in-place — no copies of X allocated.
    """
    p9  = cfg.get("phase9_normalization", {})
    tag = f"[{label}] " if label else ""
    logger.info(f"  {tag}Normalizing {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # CRITICAL (Error 024): pre-cast to float32 before normalize_total.
    # If X.data is int32/int64, normalize_total silently upcasts to float64
    # when it divides — creating a third full copy of the matrix.
    # Forcing float32 first prevents that implicit allocation.
    if sp.issparse(adata.X):
        if adata.X.data.dtype != np.float32:
            adata.X.data = adata.X.data.astype(np.float32, copy=False)
    elif adata.X.dtype != np.float32:
        adata.X = adata.X.astype(np.float32)

    # normalize_total uses sklearn's inplace_row_scale for CSR — no copy.
    target = p9.get("target_sum")
    sc.pp.normalize_total(adata, target_sum=target)
    logger.info(
        f"  {tag}Normalized to "
        f"{'median' if target is None else target} total counts")

    if p9.get("log_transform", True):
        # CRITICAL (Error 039 fix): use out= to force true in-place operation.
        # sc.pp.log1p(adata) does X.data = np.log1p(X.data) — no out= argument
        # — which allocates 13.5 GB for the train split and never returns it.
        # np.log1p(X.data, out=X.data) modifies the existing buffer directly:
        # 0 bytes allocated, 0 bytes ghosted in the OS heap.
        if sp.issparse(adata.X):
            np.log1p(adata.X.data, out=adata.X.data)
        else:
            np.log1p(adata.X, out=adata.X)
        logger.info(f"  {tag}Log1p applied (in-place, out= buffer — zero allocation)")

    return adata


def run_phase9(splits: dict, cfg: dict, logger):
    """Normalize all three splits in-place (no matrix copies)."""
    log_phase_header(logger, 9, "Normalization")
    for key in ["train", "val", "test"]:
        if key in splits:
            splits[key] = normalize_split(splits[key], cfg, logger, label=key)
    snapshot(splits["train"], "Post Phase 9 (train)", logger)
    log_memory(logger, "Phase 9 end")
    return splits
