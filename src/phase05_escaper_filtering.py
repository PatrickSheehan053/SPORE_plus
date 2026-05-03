"""
SPORE+ · src/phase05_escaper_filtering.py
──────────────────────────────────────────
Phase 5: Escaper Filtering

KEY FIX vs original SPORE phase2:
  CRISPRa DIRECTION BUG — original code always kept cells BELOW the 10th
  percentile of controls. For CRISPRa (activation), successful cells express
  the target gene MORE than controls, so the filter must be INVERTED.

  This fix reads cfg['dataset']['perturbation_type'] and branches:
    CRISPRi / CRISPRko → keep cells ≤ percentile   (knockdown = less expression)
    CRISPRa            → keep cells ≥ (100-percentile) (activation = more expression)
    mixed              → per-guide directional logic (future work)

COMBINATORIAL FIX:
  Parses compound labels like "GENE_A+GENE_B" to check knockdown of
  all constituent genes, not just a literal lookup of "GENE_A+GENE_B"
  in var_names (which would always fail and skip the filter silently).

MEMORY SAFETY:
  All lessons from SPORE error log errors 002-007 are preserved:
  - No joblib multiprocessing (uses threading only)
  - No tocsc() on full matrix (pre-caches only control cells)
  - No obs_vector() (uses direct CSR row slicing)
  - C-buffer in-place row mutation for the final subset step
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import gc

from .utils import (log_phase_header, snapshot, log_memory,
                    force_gc, safe_in_memory_row_subset)


# ═══════════════════════════════════════════════════════════════════════════
#  ESCAPER DIRECTION HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _get_escaper_direction(cfg: dict, logger) -> str:
    """
    Determine whether to keep cells with LOW or HIGH target expression.

    Returns: "knockdown" | "activation"
    """
    p5_dir  = cfg.get("phase5_escaper_filtering", {}).get("direction", "auto")
    pert_type = cfg.get("dataset", {}).get("perturbation_type", "CRISPRi")

    if p5_dir == "auto":
        if "CRISPRa" in str(pert_type) or pert_type.lower() == "crispra":
            direction = "activation"
        else:
            direction = "knockdown"
    elif p5_dir == "activation":
        direction = "activation"
    else:
        direction = "knockdown"

    logger.info(
        f"  Escaper direction: {direction} "
        f"(perturbation_type={pert_type}, config.direction={p5_dir})")
    return direction


# ═══════════════════════════════════════════════════════════════════════════
#  COMBINATORIAL LABEL PARSING
# ═══════════════════════════════════════════════════════════════════════════

def _parse_target_genes(label: str, cfg: dict) -> list:
    """
    For a perturbation label, return the list of constituent target genes.

    For single perturbations: returns [label].
    For compound labels ("GENE_A+GENE_B"): returns ["GENE_A", "GENE_B"].
    """
    if cfg.get("dataset", {}).get("perturbation_structure") == "combinatorial":
        sep = cfg.get("dataset", {}).get("perturbation_separator", "+")
        if sep and sep in str(label):
            return [p.strip() for p in str(label).split(sep) if p.strip()]
    return [str(label)]


# ═══════════════════════════════════════════════════════════════════════════
#  ESCAPER FILTER
# ═══════════════════════════════════════════════════════════════════════════

def filter_escapers(adata, cfg: dict, logger):
    """
    For each perturbation, compare target gene expression in perturbed cells
    against the control distribution.

    For CRISPRi/ko: cells with target expression > percentile of controls
    are identified as "escapers" (failed to knock down) and removed.

    For CRISPRa: cells with target expression < (100-percentile) of controls
    are identified as "escapers" (failed to activate) and removed.
    """
    log_phase_header(logger, 5, "Escaper Filtering")
    p5          = cfg.get("phase5_escaper_filtering", {})
    pert_col    = cfg.get("dataset", {}).get("perturbation_col", "gene")
    ctrl_label  = cfg.get("dataset", {}).get("control_label", "non-targeting")
    percentile  = p5.get("escaper_percentile", 10)
    direction   = _get_escaper_direction(cfg, logger)

    pert_values  = adata.obs[pert_col].values
    ctrl_mask    = pert_values == ctrl_label
    ctrl_indices = np.where(ctrl_mask)[0]
    n_ctrl       = len(ctrl_indices)
    logger.info(f"  Control cells ('{ctrl_label}'): {n_ctrl:,}")

    perturbations = [p for p in np.unique(pert_values) if p != ctrl_label]
    logger.info(f"  Unique perturbation targets: {len(perturbations):,}")

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    # Pre-cache ONLY control cells in CSC format
    # CRITICAL (Error 003): tocsc() on the FULL matrix = 2x RAM spike.
    # We only need control cell columns, so we extract only those rows first.
    logger.info(f"  Pre-caching {n_ctrl:,} control cells in CSC format...")
    ctrl_X_csc = adata.X[ctrl_indices, :].tocsc()
    log_memory(logger, "after control CSC cache")

    cells_to_keep = list(ctrl_indices)
    escaper_records = []
    n_skipped   = 0
    n_filtered  = 0
    n_comb_skip = 0

    for target in perturbations:
        pert_indices = np.where(pert_values == target)[0]
        n_total      = len(pert_indices)

        # Parse constituent genes (handles combinatorial labels)
        constituents = _parse_target_genes(target, cfg)
        genes_in_matrix = [g for g in constituents if g in gene_to_idx]

        if not genes_in_matrix:
            # No constituent gene found in var_names → bypass (silent skip)
            escaper_records.append({
                "perturbation": target, "n_total": n_total,
                "n_escaped": 0, "n_kept": n_total,
                "status": "bypassed_no_gene_in_features",
            })
            cells_to_keep.extend(pert_indices)
            n_skipped += 1
            continue

        # For combinatorial perturbations: a cell passes if ALL constituent
        # genes are knocked down (for CRISPRi) or activated (for CRISPRa).
        # Start with all-True mask; AND with each gene's threshold result.
        pass_mask = np.ones(len(pert_indices), dtype=bool)

        for gene in genes_in_matrix:
            gene_idx = gene_to_idx[gene]

            # Extract control expression for this gene
            # CRITICAL (Error 005): use pre-cached CSC column slice, NOT obs_vector
            ctrl_expr = ctrl_X_csc[:, gene_idx].toarray().flatten()

            # Extract perturbed cell expression via direct CSR row slicing
            # CRITICAL (Error 005): adata.X[pert_indices, :] then col slice
            pert_expr = adata.X[pert_indices, :][:, gene_idx]
            if sp.issparse(pert_expr):
                pert_expr = pert_expr.toarray().flatten()
            else:
                pert_expr = np.asarray(pert_expr).flatten()

            # Directional threshold
            if direction == "knockdown":
                threshold = np.percentile(ctrl_expr, percentile)
                gene_pass = pert_expr <= threshold
            else:  # activation
                threshold = np.percentile(ctrl_expr, 100 - percentile)
                gene_pass = pert_expr >= threshold

            pass_mask &= gene_pass

        kept        = pert_indices[pass_mask]
        n_escaped   = n_total - len(kept)
        cells_to_keep.extend(kept)

        status = "filtered"
        if len(constituents) > 1:
            status = "filtered_combinatorial"
            n_comb_skip += 1

        escaper_records.append({
            "perturbation": target,
            "n_total":      n_total,
            "n_escaped":    n_escaped,
            "n_kept":       len(kept),
            "n_genes_checked": len(genes_in_matrix),
            "status":       status,
        })
        n_filtered += 1

        del ctrl_expr, pert_expr, pass_mask

    del ctrl_X_csc
    gc.collect()

    escaper_stats = pd.DataFrame(escaper_records)
    total_escaped = escaper_stats["n_escaped"].sum()
    logger.info(
        f"  Filter applied to {n_filtered:,} perturbations "
        f"({n_skipped:,} bypassed — target not in features, "
        f"{n_comb_skip:,} combinatorial)")
    logger.info(f"  Total escapers removed: {total_escaped:,}")

    keep_mask = np.zeros(adata.n_obs, dtype=bool)
    keep_mask[cells_to_keep] = True

    # C-buffer in-place row mutation (Error 006/007 fix)
    adata_new = safe_in_memory_row_subset(adata, keep_mask, logger)
    snapshot(adata_new, "Post escaper filter", logger)
    return adata_new, escaper_stats


# ═══════════════════════════════════════════════════════════════════════════
#  PERTURBATION SIZE TRIAGE
# ═══════════════════════════════════════════════════════════════════════════

def filter_undersized_perturbations(adata, cfg: dict, logger):
    """
    Remove perturbation groups smaller than the minimum threshold.
    Controls are always kept regardless of cell count.
    """
    p5         = cfg.get("phase5_escaper_filtering", {})
    pert_col   = cfg.get("dataset", {}).get("perturbation_col", "gene")
    ctrl_label = cfg.get("dataset", {}).get("control_label", "non-targeting")
    min_cells  = p5.get("min_cells_per_perturbation", 50)

    sizes       = adata.obs[pert_col].value_counts()
    undersized  = sizes[(sizes < min_cells) & (sizes.index != ctrl_label)]

    n_drop_perts = len(undersized)
    n_drop_cells = undersized.sum()

    if n_drop_perts > 0:
        drop_labels = set(undersized.index)
        keep_mask   = ~adata.obs[pert_col].isin(drop_labels).values
        logger.info(
            f"  Perturbation triage (min {min_cells} cells): "
            f"dropped {n_drop_perts:,} perturbations ({n_drop_cells:,} cells)")

        adata = safe_in_memory_row_subset(adata, keep_mask, logger)
    else:
        logger.info(
            f"  Perturbation triage: all groups meet minimum ({min_cells} cells)")

    final_sizes = adata.obs[pert_col].value_counts()
    snapshot(adata, "Post perturbation triage", logger)
    return adata, final_sizes


def run_phase5(adata, cfg: dict, logger):
    """Full Phase 5: escaper filtering → perturbation triage."""
    adata, escaper_stats = filter_escapers(adata, cfg, logger)
    adata, pert_sizes    = filter_undersized_perturbations(adata, cfg, logger)
    return adata, escaper_stats, pert_sizes
