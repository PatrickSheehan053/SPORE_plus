"""
SPORE+ · src/phase05_escaper_filtering.py
──────────────────────────────────────────
Phase 5: Escaper Filtering + Knockdown Efficiency Scoring (Milestone 2)

MILESTONE 2 UPGRADE — Efficiency Scoring:
  After the escaper filter, for each perturbation group compute:
    efficiency_score        : fraction of cells that passed the filter
    mean_knockdown_depth    : mean target expression in kept cells / mean in controls
                              (lower = better knockdown, 0 = perfect, 1 = no effect)
    low_efficiency_flag     : True if efficiency_score < efficiency_threshold

  These are stored in adata.uns["knockdown_efficiency"] and added as per-cell
  obs columns: 'escaper_efficiency', 'escaper_knockdown_depth'.

  This gives a per-perturbation quality readout that CHITIN can weight downstream.

CRISPRa DIRECTION FIX (Milestone 1):
  For knockdown (CRISPRi/ko): keep cells ≤ Nth percentile of controls
  For activation (CRISPRa): keep cells ≥ (100-N)th percentile of controls

COMBINATORIAL FIX (Milestone 1):
  "GENE_A+GENE_B" labels are parsed to constituent genes for threshold comparison.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import gc

from .utils import (log_phase_header, snapshot, log_memory,
                    force_gc, safe_in_memory_row_subset)


def _get_escaper_direction(cfg: dict, logger) -> str:
    p5_dir    = cfg.get("phase5_escaper_filtering", {}).get("direction", "auto")
    pert_type = cfg.get("dataset", {}).get("perturbation_type", "CRISPRi")
    if p5_dir == "auto":
        direction = "activation" if "CRISPRa" in str(pert_type) else "knockdown"
    else:
        direction = "activation" if p5_dir == "activation" else "knockdown"
    logger.info(f"  Direction: {direction} (perturbation_type={pert_type})")
    return direction


def _parse_target_genes(label: str, cfg: dict) -> list:
    if cfg.get("dataset", {}).get("perturbation_structure") == "combinatorial":
        sep = cfg.get("dataset", {}).get("perturbation_separator", "+")
        if sep and sep in str(label):
            return [p.strip() for p in str(label).split(sep) if p.strip()]
    return [str(label)]


def filter_escapers(adata, cfg: dict, logger):
    log_phase_header(logger, 5, "Escaper Filtering + Efficiency Scoring")
    p5         = cfg.get("phase5_escaper_filtering", {})
    pert_col   = cfg.get("dataset", {}).get("perturbation_col", "gene")
    ctrl_label = cfg.get("dataset", {}).get("control_label", "non-targeting")
    percentile = p5.get("escaper_percentile", 10)
    direction  = _get_escaper_direction(cfg, logger)
    eff_thresh = p5.get("efficiency_threshold", 0.50)

    pert_values  = adata.obs[pert_col].values
    ctrl_mask    = pert_values == ctrl_label
    ctrl_indices = np.where(ctrl_mask)[0]
    n_ctrl       = len(ctrl_indices)
    logger.info(f"  Control cells ('{ctrl_label}'): {n_ctrl:,}")

    perturbations = [p for p in np.unique(pert_values) if p != ctrl_label]
    logger.info(f"  Perturbation targets: {len(perturbations):,}")

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    # Pre-cache control cells in CSC for efficient column slicing (Error 003 fix)
    ctrl_X_csc = adata.X[ctrl_indices, :].tocsc()
    log_memory(logger, "after control CSC cache")

    cells_to_keep   = list(ctrl_indices)
    escaper_records = []
    efficiency_records = {}   # pert → {efficiency, knockdown_depth, low_efficiency}
    n_skipped = 0

    for target in perturbations:
        pert_indices = np.where(pert_values == target)[0]
        n_total      = len(pert_indices)
        constituents = _parse_target_genes(target, cfg)
        genes_in_mat = [g for g in constituents if g in gene_to_idx]

        if not genes_in_mat:
            escaper_records.append({
                "perturbation": target, "n_total": n_total,
                "n_escaped": 0, "n_kept": n_total,
                "status": "bypassed_no_gene_in_features"})
            cells_to_keep.extend(pert_indices)
            efficiency_records[target] = {
                "efficiency_score": 1.0, "knockdown_depth": float("nan"),
                "low_efficiency": False, "n_genes_checked": 0}
            n_skipped += 1
            continue

        pass_mask = np.ones(len(pert_indices), dtype=bool)
        knockdown_depths = []

        for gene in genes_in_mat:
            gene_idx  = gene_to_idx[gene]
            ctrl_expr = ctrl_X_csc[:, gene_idx].toarray().flatten()
            pert_expr = adata.X[pert_indices, :][:, gene_idx]
            if sp.issparse(pert_expr):
                pert_expr = pert_expr.toarray().flatten()
            else:
                pert_expr = np.asarray(pert_expr).flatten()

            if direction == "knockdown":
                threshold = np.percentile(ctrl_expr, percentile)
                gene_pass = pert_expr <= threshold
                ctrl_mean = ctrl_expr.mean()
                if ctrl_mean > 1e-10:
                    kept_mean = pert_expr[gene_pass].mean() if gene_pass.any() else 0
                    knockdown_depths.append(kept_mean / ctrl_mean)
            else:
                threshold = np.percentile(ctrl_expr, 100 - percentile)
                gene_pass = pert_expr >= threshold
                ctrl_mean = ctrl_expr.mean()
                if ctrl_mean > 1e-10:
                    kept_mean = pert_expr[gene_pass].mean() if gene_pass.any() else 0
                    knockdown_depths.append(kept_mean / ctrl_mean)

            pass_mask &= gene_pass

        kept      = pert_indices[pass_mask]
        n_escaped = n_total - len(kept)
        efficiency = len(kept) / max(n_total, 1)
        kd_depth   = float(np.mean(knockdown_depths)) if knockdown_depths else float("nan")

        cells_to_keep.extend(kept)
        escaper_records.append({
            "perturbation": target, "n_total": n_total,
            "n_escaped": n_escaped, "n_kept": len(kept),
            "n_genes_checked": len(genes_in_mat),
            "status": "filtered"})
        efficiency_records[target] = {
            "efficiency_score":  efficiency,
            "knockdown_depth":   kd_depth,
            "low_efficiency":    efficiency < eff_thresh}

        del ctrl_expr, pert_expr, pass_mask

    del ctrl_X_csc
    gc.collect()

    # Build per-cell efficiency columns
    eff_scores = np.ones(adata.n_obs, dtype=np.float32)
    kd_depths  = np.full(adata.n_obs, float("nan"), dtype=np.float32)
    for i, pv in enumerate(pert_values):
        if pv in efficiency_records:
            eff_scores[i] = efficiency_records[pv]["efficiency_score"]
            kd_val = efficiency_records[pv]["knockdown_depth"]
            if not np.isnan(kd_val):
                kd_depths[i] = kd_val

    escaper_stats = pd.DataFrame(escaper_records)
    total_escaped = escaper_stats["n_escaped"].sum()
    n_low_eff     = sum(1 for r in efficiency_records.values() if r["low_efficiency"])
    logger.info(f"  Escapers removed: {total_escaped:,}  |  {n_skipped:,} bypassed (gene not in features)")
    logger.info(f"  Efficiency: {n_low_eff:,} perturbations below {eff_thresh*100:.0f}% threshold")

    keep_mask = np.zeros(adata.n_obs, dtype=bool)
    keep_mask[cells_to_keep] = True

    adata_new = safe_in_memory_row_subset(adata, keep_mask, logger)

    # Store efficiency metadata
    adata_new.obs["escaper_efficiency"]     = eff_scores[keep_mask]
    adata_new.obs["escaper_knockdown_depth"] = kd_depths[keep_mask]
    adata_new.uns["knockdown_efficiency"]   = {
        k: {kk: (float(vv) if isinstance(vv, (float, np.floating))
                 else bool(vv) if isinstance(vv, (bool, np.bool_))
                 else vv)
            for kk, vv in v.items()}
        for k, v in efficiency_records.items()}

    snapshot(adata_new, "Post escaper filter", logger)
    return adata_new, escaper_stats


def filter_undersized_perturbations(adata, cfg: dict, logger):
    p5        = cfg.get("phase5_escaper_filtering", {})
    pert_col  = cfg.get("dataset", {}).get("perturbation_col", "gene")
    ctrl_label = cfg.get("dataset", {}).get("control_label", "non-targeting")
    min_cells  = p5.get("min_cells_per_perturbation", 50)

    sizes      = adata.obs[pert_col].value_counts()
    undersized = sizes[(sizes < min_cells) & (sizes.index != ctrl_label)]
    if len(undersized) > 0:
        drop_labels = set(undersized.index)
        keep_mask   = ~adata.obs[pert_col].isin(drop_labels).values
        logger.info(f"  Perturbation triage (min {min_cells}): "
                    f"dropped {len(undersized):,} groups ({undersized.sum():,} cells)")
        adata = safe_in_memory_row_subset(adata, keep_mask, logger)
    else:
        logger.info(f"  Perturbation triage: all groups meet minimum ({min_cells})")

    final_sizes = adata.obs[pert_col].value_counts()
    snapshot(adata, "Post perturbation triage", logger)
    return adata, final_sizes


def run_phase5(adata, cfg: dict, logger):
    adata, escaper_stats = filter_escapers(adata, cfg, logger)
    adata, pert_sizes    = filter_undersized_perturbations(adata, cfg, logger)
    return adata, escaper_stats, pert_sizes
