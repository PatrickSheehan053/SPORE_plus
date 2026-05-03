"""
SPORE+ · src/phase02_cell_triage.py
─────────────────────────────────────
Phase 2: Cell-Level Triage
Adapted from SPORE phase1_cell_triage.py.
Config key: phase2_cell_triage (was phase1_cell_triage).
snRNA-seq guard: if _detection_overrides.disable_mt_filter is True,
MT% threshold is set to 1.0 (effectively disabled).
"""

import numpy as np
import scanpy as sc
from collections import OrderedDict
from .utils import (log_phase_header, snapshot, ensure_sparse,
                    safe_in_memory_row_subset, safe_in_memory_gene_subset,
                    log_memory, force_gc)


def _ensure_qc_metrics(adata, cfg, logger):
    """
    QC METRIC FIREWALL (Error 026/027 fix):
    Raw matrices don't have total_counts / pct_counts_mt.
    Both plotting and filter_cells call this — earliest touch wins.
    """
    if "total_counts" not in adata.obs.columns or \
       "pct_counts_mt" not in adata.obs.columns:
        logger.info(
            "  QC Metric Firewall: raw matrix detected, "
            "calculating total_counts and pct_counts_mt...")
        organism = cfg.get("dataset", {}).get("organism", "human")
        mt_prefix = "MT-" if organism == "human" else "mt-"
        adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], percent_top=None,
            log1p=False, inplace=True)
    return adata


def filter_cells(adata, cfg: dict, logger):
    log_phase_header(logger, 2, "Cell-Level Triage")
    adata = ensure_sparse(adata, logger)
    adata = _ensure_qc_metrics(adata, cfg, logger)

    p2  = cfg.get("phase2_cell_triage", {})
    overrides = cfg.get("_detection_overrides", {})
    waterfall = OrderedDict()
    waterfall["Starting cells"] = adata.n_obs

    keep_cells = np.ones(adata.n_obs, dtype=bool)

    # ── MT filter ──────────────────────────────────────────────────────────
    # snRNA-seq guard: disable MT filter if Phase 1 flagged snRNA-seq
    disable_mt = overrides.get("disable_mt_filter", False)
    if disable_mt:
        logger.info(
            "  MT filter: DISABLED (snRNA-seq mode — no mitochondrial genes in nuclei)")
    else:
        mt_thresh = p2.get("mt_threshold", 0.20)
        before = keep_cells.sum()
        keep_cells &= (adata.obs["pct_counts_mt"].values <= mt_thresh * 100)
        logger.info(
            f"  MT filter (≤{mt_thresh*100:.0f}%): "
            f"removed {before - keep_cells.sum():,} cells")
    waterfall["After MT filter"] = int(keep_cells.sum())

    # ── Gene count bounds ──────────────────────────────────────────────────
    before = keep_cells.sum()
    keep_cells &= (
        adata.obs["n_genes_by_counts"].values >= p2["min_genes_per_cell"])
    logger.info(
        f"  Min genes (≥{p2['min_genes_per_cell']}): "
        f"removed {before - keep_cells.sum():,}")
    waterfall["After min genes"] = int(keep_cells.sum())

    before = keep_cells.sum()
    keep_cells &= (
        adata.obs["n_genes_by_counts"].values <= p2["max_genes_per_cell"])
    logger.info(
        f"  Max genes (≤{p2['max_genes_per_cell']:,}): "
        f"removed {before - keep_cells.sum():,}")
    waterfall["After max genes"] = int(keep_cells.sum())

    # ── UMI count bounds ───────────────────────────────────────────────────
    before = keep_cells.sum()
    keep_cells &= (
        adata.obs["total_counts"].values >= p2["min_counts_per_cell"])
    logger.info(
        f"  Min counts (≥{p2['min_counts_per_cell']}): "
        f"removed {before - keep_cells.sum():,}")
    waterfall["After min counts"] = int(keep_cells.sum())

    before = keep_cells.sum()
    keep_cells &= (
        adata.obs["total_counts"].values <= p2["max_counts_per_cell"])
    logger.info(
        f"  Max counts (≤{p2['max_counts_per_cell']:,}): "
        f"removed {before - keep_cells.sum():,}")
    waterfall["After max counts"] = int(keep_cells.sum())

    # ── Ribosomal gene removal ─────────────────────────────────────────────
    ribo_prefixes = tuple(p2.get("ribo_prefixes", ["RPL", "RPS"]))
    keep_genes = ~adata.var_names.str.startswith(ribo_prefixes)
    n_ribo = (~keep_genes).sum()
    logger.info(
        f"  Ribosomal gene removal: {n_ribo:,} genes stripped")

    log_memory(logger, "before cell subset")
    adata_new = safe_in_memory_row_subset(adata, keep_cells, logger)
    del adata
    force_gc(logger)

    # Apply gene mask (ribosomal removal)
    adata_new = safe_in_memory_gene_subset(adata_new, keep_genes, logger)

    snapshot(adata_new, "Post Phase 2", logger)
    return adata_new, waterfall


def run_phase2(adata, cfg: dict, logger):
    return filter_cells(adata, cfg, logger)
