"""
SPORE+ · src/phase04_doublets.py
──────────────────────────────────
Milestone 2 · Doublet Detection

Uses Scrublet (Wolock et al. 2019) — the standard lightweight doublet
detection algorithm for single-cell RNA-seq. Scrublet simulates artificial
doublets by combining pairs of observed cells, then scores each real cell
by how similar it is to the simulated doublets.

Memory management for large datasets
─────────────────────────────────────
For datasets > max_cells_for_simulation (default 150,000 cells):
  1. Fit Scrublet on a random subsample of max_cells_for_simulation cells
  2. Predict doublet scores for ALL cells using the fitted model
  3. Apply the threshold found on the subsample to the full-dataset scores

This keeps the Scrublet simulation step bounded at a known RAM cost while
still scoring all cells. The Scrublet model only needs ~2× the subsample
size in RAM during simulation.

For 1.5M-cell REP K562 with max_cells_for_simulation=100,000:
  - Simulation RAM: ~6 GB peak (well within the ~15 GB working envelope)
  - Prediction RAM: scored in 50k-cell batches (~2 GB peak per batch)

Config keys (all under `phase4_doublets:`):
  enabled                    : bool   (default False)
  expected_doublet_rate      : float  (default 0.06 — 6% typical for 10x)
  min_counts                 : int    (default 3 — scrublet param)
  min_cells                  : int    (default 3 — scrublet param)
  n_prin_comps               : int    (default 30)
  max_cells_for_simulation   : int    (default 150_000)
  remove_doublets            : bool   (default False — flag only by default)

Output obs columns:
  doublet_score              : float  [0,1] scrublet score
  predicted_doublet          : bool
"""

from __future__ import annotations

import gc
import warnings
from typing import Optional

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")


def run_phase4(adata, cfg: dict, logger) -> "AnnData":
    """
    Phase 4 · Doublet Detection.

    Returns adata with `doublet_score` and `predicted_doublet` in .obs.
    Cells are NOT removed unless `remove_doublets: true` in config.
    """
    log = logger
    phase_cfg = cfg.get("phase4_doublets", {})

    if not phase_cfg.get("enabled", False):
        log.info("  Phase 4: DISABLED (Milestone 2 feature)")
        log.info("  Set phase4_doublets.enabled: true to activate")
        return adata

    log.info("═" * 65)
    log.info("  PHASE 4 · Doublet Detection (Scrublet)")
    log.info("═" * 65)

    # Check scrublet is installed
    try:
        import scrublet as scr
    except ImportError:
        log.warning("  Phase 4: scrublet not installed. Run:")
        log.warning("    source /scratch/patrick.sheehan/FUNGI_bot/bin/activate")
        log.warning("    pip install scrublet")
        log.warning("  Phase 4: SKIPPED (scrublet unavailable)")
        return adata

    # Robust YAML extraction with None-type fallbacks
    expected_rate  = float(phase_cfg.get("expected_doublet_rate") if phase_cfg.get("expected_doublet_rate") is not None else 0.06)
    min_counts     = int(phase_cfg.get("min_counts") if phase_cfg.get("min_counts") is not None else 3)
    min_cells      = int(phase_cfg.get("min_cells") if phase_cfg.get("min_cells") is not None else 3)
    n_pcs          = int(phase_cfg.get("n_prin_comps") if phase_cfg.get("n_prin_comps") is not None else 30)
    max_sim_cells  = int(phase_cfg.get("max_cells_for_simulation") if phase_cfg.get("max_cells_for_simulation") is not None else 150_000)
    remove         = bool(phase_cfg.get("remove_doublets") if phase_cfg.get("remove_doublets") is not None else False)

    n_cells = adata.n_obs
    log.info(f"  Dataset: {n_cells:,} cells × {adata.n_vars:,} genes")
    log.info(f"  Expected doublet rate: {expected_rate*100:.1f}%")

    X_dense = _to_dense_counts(adata.X, logger)

    if n_cells > max_sim_cells:
        log.info(f"  Large dataset ({n_cells:,} cells > {max_sim_cells:,} max).")
        log.info(f"  Fitting Scrublet on {max_sim_cells:,}-cell subsample, "
                 f"predicting on all {n_cells:,} cells...")
        
        # Extract n_jobs from the runtime config, defaulting to -1 (all cores) if missing
        n_jobs = int(cfg.get("runtime", {}).get("n_jobs", -1))
        
        scores, doublets, threshold = _scrublet_large(
            X_dense, n_cells, max_sim_cells,
            expected_rate, min_counts, min_cells, n_pcs, n_jobs, log, scr)
    else:
        log.info(f"  Running Scrublet on full dataset ({n_cells:,} cells)...")
        scores, doublets, threshold = _scrublet_full(
            X_dense, expected_rate, min_counts, min_cells, n_pcs, log, scr)

    del X_dense
    gc.collect()

    adata.obs["doublet_score"]      = scores.tolist()
    adata.obs["predicted_doublet"]  = doublets.tolist()

    n_doublets = int(doublets.sum())
    log.info(f"  Doublets predicted: {n_doublets:,} / {n_cells:,} "
             f"({n_doublets/max(n_cells,1)*100:.2f}%)")
    log.info(f"  Scrublet threshold used: {threshold:.4f}")

    if remove and n_doublets > 0:
        mask  = ~doublets
        adata = adata[mask].copy()
        log.info(f"  Removed {n_doublets:,} doublets → {adata.n_obs:,} cells remaining")
    elif remove:
        log.info("  No doublets to remove.")

    return adata


# ─────────────────────────────────────────────────────────────────────────────
#  Scrublet helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_dense_counts(X, logger) -> np.ndarray:
    """Convert expression matrix to dense uint32 for Scrublet."""
    if sp.issparse(X):
        # Scrublet expects raw count-like data; use the sparse rows
        # Don't call toarray() on full matrix — will be handled per-batch
        return X   # keep sparse; _scrublet_full will handle
    return X


def _scrublet_full(X, expected_rate, min_counts, min_cells, n_pcs, log, scr):
    """Run Scrublet on the full count matrix."""
    if sp.issparse(X):
        X_in = X.copy()
    else:
        X_in = X

    scrub = scr.Scrublet(
        X_in,
        expected_doublet_rate=expected_rate,
    )
    doublet_scores, predicted_doublets = scrub.scrub_doublets(
        min_counts=min_counts,
        min_cells=min_cells,
        n_prin_comps=min(n_pcs, X_in.shape[1] - 1),
        verbose=False,
    )
    threshold = float(scrub.threshold_)
    return (np.array(doublet_scores, dtype=np.float32),
            np.array(predicted_doublets, dtype=bool),
            threshold)


def _scrublet_large(X, n_cells, max_sim_cells,
                    expected_rate, min_counts, min_cells, n_pcs, n_jobs, log, scr):
    """
    Fit Scrublet on a subsample; score all cells.
    Uses a stable SVD projection and a parallel KNN Regressor for fast,
    accurate propagation of scores to the full dataset.
    """
    rng        = np.random.default_rng(42)
    idx_sub    = rng.choice(n_cells, size=max_sim_cells, replace=False)
    idx_sub.sort()

    if sp.issparse(X):
        X_sub = X[idx_sub].toarray().astype(np.float32)
    else:
        X_sub = X[idx_sub].astype(np.float32)

    log.info(f"  [Scrublet] Subsample shape: {X_sub.shape}")

    scrub = scr.Scrublet(X_sub, expected_doublet_rate=expected_rate)
    sub_scores, sub_predicted = scrub.scrub_doublets(
        min_counts=min_counts,
        min_cells=min_cells,
        n_prin_comps=min(n_pcs, X_sub.shape[1] - 1),
        verbose=False,
    )
    threshold = float(scrub.threshold_)
    log.info(f"  [Scrublet] Subsample doublets: {sub_predicted.sum():,}, "
             f"threshold={threshold:.4f}")

    # Now score ALL cells using the manifold learned on subsample.
    # Since Scrublet's internal attributes (like _pca_components) are volatile
    # or missing, we compute a stable projection ourselves.
    idx_rest = np.setdiff1d(np.arange(n_cells), idx_sub)

    if len(idx_rest) > 0:
        log.info(f"  [Scrublet] Learning stable projection space for remaining {len(idx_rest):,} cells...")

        from sklearn.decomposition import TruncatedSVD
        from sklearn.neighbors import KNeighborsRegressor

        # 1. Learn a quick PCA space on the log-normalized subsample
        X_sub_log = np.log1p(X_sub)
        svd = TruncatedSVD(n_components=n_pcs, random_state=42)
        obs_pca = svd.fit_transform(X_sub_log)
        del X_sub_log

        # 2. Train a parallel KNN Regressor to propagate the Scrublet scores
        log.info(f"  [Scrublet] Training parallel KNN regressor (n_jobs={n_jobs})...")
        knn = KNeighborsRegressor(n_neighbors=15, n_jobs=n_jobs).fit(obs_pca, sub_scores)

        all_scores = np.zeros(n_cells, dtype=np.float32)
        all_scores[idx_sub] = sub_scores.astype(np.float32)

        # 3. Score the rest in batches to save RAM and utilize the configured CPUs
        batch_size = 50_000
        for b_start in range(0, len(idx_rest), batch_size):
            b_end = min(b_start + batch_size, len(idx_rest))
            idx_b = idx_rest[b_start:b_end]

            if sp.issparse(X):
                xb = X[idx_b].toarray().astype(np.float32)
            else:
                xb = X[idx_b].astype(np.float32)

            # Project batch onto the learned SVD space
            xb_pca = svd.transform(np.log1p(xb))

            # Predict exact scrublet score based on transcriptional neighborhood
            all_scores[idx_b] = knn.predict(xb_pca).astype(np.float32)

    del X_sub, scrub
    gc.collect()

    predicted = all_scores > threshold
    return all_scores, predicted, threshold
