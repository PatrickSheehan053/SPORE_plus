"""
SPORE+ · src/phase08_hvg.py
──────────────────────────────
Phase 8: Dimensionality Reduction (Highly Variable Genes)

Engineered for zero-leakage Foundation Model training. 
Unlike standard single-cell pipelines that calculate HVGs globally, 
SPORE+ strictly computes variance metrics on the TRAINING split only, 
and projects that feature mask onto the Validation and Test splits.

Stratified Subsampling & Memory Safety
──────────────────────────────────────
The `seurat_v3` variance flavor natively requires dense matrix math. To 
prevent OOM crashes on 1M+ cell datasets, the calculation is performed 
on a 100k-cell subsample. To ensure rare perturbations aren't erased 
by random chance, this subsample is strictly proportionally stratified 
across all guide RNA groups.

Rescue Operations
─────────────────
Perturbation targets and cell cycle genes (Ghosts) are forcibly injected 
into the final HVG set if they were missed by the variance calculation, 
ensuring they remain available for downstream regression and MSE loss evaluation.
"""

import numpy as np
import scipy.sparse as sp
import scanpy as sc
from .utils import (log_phase_header, snapshot, log_memory, force_gc,
                    safe_in_memory_gene_subset, get_cell_cycle_genes)


def select_hvgs(adata, cfg: dict, logger, split_label: str = "train"):
    # --- SAFETY CATCH ---
    if "hvg_stats" in adata.uns:
        logger.warning(f"  ⚠ Phase 8 has already been run on {split_label}! Skipping to prevent double-subsetting.")
        return adata, list(adata.var_names)
    # --------------------
    """
    Select highly variable genes on the training split to prevent data leakage.
    Uses 100k-cell STRATIFIED subsample for seurat_v3 variance calculation.
    Saves HVG stats to .uns['hvg_stats'] BEFORE subsetting the matrix.
    """
    import pandas as pd
    
    log_phase_header(logger, 8, f"Dimensionality Reduction – HVG ({split_label})")
    p8   = cfg.get("phase8_hvg", {})
    n_top = p8.get("n_top_genes", 5000)
    method = p8.get("method", "seurat_v3")

    if not p8.get("enabled", True):
        logger.info("  Phase 8: HVG selection DISABLED")
        return adata, []

    log_memory(logger, "Phase 8 start")

    # ── Ghost re-rescue at Phase 8 ─────────────────────────────────────────
    cc_s, cc_g2m = get_cell_cycle_genes()
    cc_genes  = set(cc_s + cc_g2m)
    var_upper = {v.upper(): v for v in adata.var_names}
    cc_in_matrix = [var_upper[g] for g in cc_genes if g in var_upper]

    # ── STRATIFIED Subsample for HVG calculation (The Fix) ─────────────────
    n_sample = min(100_000, adata.n_obs)
    if n_sample < adata.n_obs:
        pert_col = cfg.get("dataset", {}).get("perturbation_col", "perturbation")
        
        if pert_col in adata.obs.columns:
            logger.info(
                f"  HVG calc: Stratified sampling ~{n_sample:,}/{adata.n_obs:,} cells "
                f"to preserve rare target variance...")
            
            frac = n_sample / adata.n_obs
            df = pd.DataFrame({"pert": adata.obs[pert_col]})
            df["orig_idx"] = np.arange(adata.n_obs)

            rng = np.random.default_rng(42)
            sampled_indices = []

            # THE FIX: Add observed=True and skip empty groups
            for pert, group in df.groupby("pert", observed=True):
                n_group = len(group)
                
                if n_group == 0:
                    continue  # Skip perturbations strictly held out in Val/Test sets

                n_take = int(np.round(n_group * frac))
                
                # CRITICAL: Force at least 1 cell for extremely rare perturbations
                n_take = max(1, min(n_take, n_group))
                idx = rng.choice(group["orig_idx"].values, size=n_take, replace=False)
                sampled_indices.extend(idx)

            idx = np.array(sampled_indices)
            # THE FIX: Removed rng.shuffle(idx). Variance is permutation invariant.
            # HDF5 strictly requires sorted index arrays to prevent I/O crashes.
        else:
            logger.warning("  Perturbation col not found. Falling back to random sample.")
            rng = np.random.default_rng(42)
            idx = rng.choice(adata.n_obs, size=n_sample, replace=False)

        # ── OOM FIREWALL: Safe Subsample Extraction ──
        idx_sorted = np.sort(idx)
        is_large = getattr(adata, 'isbacked', False) or adata.n_obs > 1000000

        if is_large:
            logger.info("  [HVG] Large/Backed mode: Extracting stratified subsample in safe micro-chunks...")
            chunks = []
            chunk_size = 10000
            for i in range(0, len(idx_sorted), chunk_size):
                sub_idx = idx_sorted[i:i+chunk_size]
                # HDF5 handles sorted index arrays sequentially, avoiding buffer bloat
                c = adata.X[sub_idx]
                if not sp.issparse(c):
                    c = sp.csr_matrix(c)
                chunks.append(c)
            calc_X = sp.vstack(chunks)
        else:
            calc_X = adata.X[idx_sorted, :]
            
        import anndata as ad
        calc_adata = ad.AnnData(X=calc_X, var=adata.var.copy())
    else:
        import anndata as ad
        calc_adata = ad.AnnData(X=adata.X.copy(), var=adata.var.copy())

    if sp.issparse(calc_adata.X):
        calc_adata.X.data = calc_adata.X.data.astype(np.float32, copy=False)

    logger.info(f"  Running HVG selection: method={method}, n_top={n_top:,}")
    sc.pp.highly_variable_genes(
        calc_adata, flavor=method,
        n_top_genes=n_top, subset=False)

    # ── Extract stats and immediately flush calc_adata to save RAM ──
    stat_cols = [c for c in calc_adata.var.columns
                 if c in ("highly_variable", "means",
                          "dispersions", "dispersions_norm",
                          "variances", "variances_norm")]
    hvg_stats = calc_adata.var[stat_cols].copy()
    hvg_names = set(calc_adata.var_names[calc_adata.var["highly_variable"]])
    rank_col = "variances_norm" if "variances_norm" in hvg_stats.columns else "dispersions_norm"
    
    del calc_adata
    force_gc(logger)

    # Initialize tracking columns for diagnostics
    hvg_stats["rescued_target"] = False
    hvg_stats["rescued_ghost"] = False

    # ── 1. Target Rescue: The 1-to-1 Swap ──────────────────────────────────
    if p8.get("rescue_perturbation_targets", True):
        pert_col   = cfg["dataset"]["perturbation_col"]
        ctrl_label = cfg["dataset"]["control_label"]
        is_comb    = cfg.get("dataset", {}).get("perturbation_structure") == "combinatorial"
        sep        = cfg.get("dataset", {}).get("perturbation_separator", "+")
        var_set    = set(adata.var_names)
        raw_labels = set(adata.obs[pert_col].unique()) - {ctrl_label}

        targets = set()
        for lbl in raw_labels:
            if is_comb and sep and sep in str(lbl):
                for part in str(lbl).split(sep):
                    targets.add(part.strip())
            else:
                targets.add(str(lbl))

        target_rescue = (targets & var_set) - hvg_names
        if target_rescue:
            n_rescue = len(target_rescue)
            logger.info(f"  ⚡ Target Rescue: {n_rescue} perturbation targets missing.")

            # Identify "pure" HVGs that are safe to drop (exclude targets/ghosts)
            safe_to_drop = list(hvg_names - targets - set(cc_in_matrix))

            if rank_col in hvg_stats.columns and len(safe_to_drop) >= n_rescue:
                # Find the lowest variance genes
                safe_scores = hvg_stats.loc[safe_to_drop, rank_col].sort_values(ascending=True)
                genes_to_drop = safe_scores.head(n_rescue).index.tolist()

                # Execute the Swap
                hvg_names.difference_update(genes_to_drop)
                hvg_names.update(target_rescue)

                # Update the stats DataFrame to reflect the swap
                hvg_stats.loc[genes_to_drop, "highly_variable"] = False
                hvg_stats.loc[list(target_rescue), "highly_variable"] = True
                hvg_stats.loc[list(target_rescue), "rescued_target"] = True

                logger.info(f"  ⚡ Swapped bottom {len(genes_to_drop)} HVGs to backfill targets (Core stabilized at {n_top:,}).")
            else:
                logger.warning("  ⚡ Could not execute 1-to-1 swap. Adding targets on top.")
                hvg_names.update(target_rescue)
                hvg_stats.loc[list(target_rescue), "rescued_target"] = True

    # ── 2. Ghost Rescue: The Stack ─────────────────────────────────────────
    ghosts = [g for g in cc_in_matrix if g not in hvg_names]
    if ghosts:
        logger.info(f"  👻 Ghost Rescue: {len(ghosts)} cell cycle genes stacked on top.")
        hvg_names.update(ghosts)
        hvg_stats.loc[ghosts, "rescued_ghost"] = True

    # ── OOM FIREWALL: Safely handle .uns mutation & subsetting ──
    is_large = getattr(adata, 'isbacked', False) or adata.n_obs > 1000000
    
    keep_mask = np.array([g in hvg_names for g in adata.var_names])
    n_kept    = keep_mask.sum()
    logger.info(f"  Subsetting from {adata.n_vars:,} → {n_kept:,} features")
    
    core_features = [g for g in adata.var_names if g in hvg_names and g not in set(ghosts)]

    if is_large:
        logger.info("  [HVG] Large/Backed mode: Bypassing .uns HDF5 mutation to prevent kernel segfault...")
        adata_new = adata[:, keep_mask]
        
        # AnnData views lock the .uns dict to prevent HDF5 corruption. 
        # We override it with a brand new independent Python dictionary in memory.
        adata_new.uns = adata.uns.copy() if hasattr(adata, 'uns') else {}
        adata_new.uns["hvg_stats"] = hvg_stats.to_dict("index")
        adata_new.uns["spore_core_features"] = core_features
        logger.info(f"  HVG stats safely saved to in-memory .uns dictionary.")
    else:
        adata.uns["hvg_stats"] = hvg_stats.to_dict("index")
        adata.uns["spore_core_features"] = core_features
        logger.info(f"  HVG stats saved to .uns['hvg_stats']")
        adata_new = safe_in_memory_gene_subset(adata, keep_mask=keep_mask, logger=logger)
        del adata
        
    force_gc(logger)

    snapshot(adata_new, "Post HVG selection", logger)
    log_memory(logger, "Phase 8 end")
    return adata_new, list(hvg_names)


def apply_hvg_to_other_splits(splits: dict, hvg_names: list,
                               cfg: dict, logger):
    """
    Apply the HVG mask determined from train to val and test splits.
    CRITICAL: HVG must be computed on train ONLY to prevent data leakage.
    The same gene set is then applied to val/test.
    """
    logger.info(f"  Applying HVG mask to val/test splits "
                f"({len(hvg_names):,} features)...")
    hvg_set = set(hvg_names)

    for key in ["val", "test"]:
        if key not in splits:
            continue
        split_adata = splits[key]
        keep_mask   = np.array([g in hvg_set for g in split_adata.var_names])
        
        # ── OOM FIREWALL: Apply via lazy views for backed objects ──
        is_large = getattr(split_adata, 'isbacked', False) or split_adata.n_obs > 1000000
        if is_large:
            splits[key] = split_adata[:, keep_mask]
        else:
            splits[key] = safe_in_memory_gene_subset(
                split_adata, keep_mask=keep_mask, logger=logger)
            
        snapshot(splits[key], f"HVG applied to {key}", logger)
        force_gc(logger)
    return splits


def run_phase8(splits: dict, cfg: dict, logger):
    """
    Full Phase 8: compute HVG on train, apply to val/test.
    Returns updated splits dict.
    """
    train_new, hvg_names = select_hvgs(
        splits["train"], cfg, logger, split_label="train")
    splits["train"] = train_new
    splits = apply_hvg_to_other_splits(splits, hvg_names, cfg, logger)
    return splits, hvg_names
