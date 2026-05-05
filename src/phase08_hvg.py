"""
SPORE+ · src/phase08_hvg.py
──────────────────────────────
Phase 8: Highly Variable Gene Selection
Adapted from SPORE phase5_hvg.py.
Config key: phase8_hvg (was phase5_hvg).

Critical memory-safety lessons from SPORE error log applied:
  Error 017: seurat_v3 X**2 OOM — use 100k subsample for HVG calc
  Error 018: HVG metadata destroyed when calc_adata deleted — save to .uns
  Error 019: seurat_v3 uses 'variances_norm' not 'dispersions_norm'
  Error 010: gene subsetting uses C-buffer row reconstructor

Ghost Rescue (Error 033 re-applied at Phase 8):
  Phase 6 rescued cell cycle genes from the sparsity cut.
  But the adaptive UMI filter in Phase 6 may have already wiped them
  before Phase 8 can include them in the HVG set.
  We re-check and re-ghost them here if they were somehow dropped.
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
            rng.shuffle(idx) # Shuffle to remove order bias
        else:
            logger.warning("  Perturbation col not found. Falling back to random sample.")
            rng = np.random.default_rng(42)
            idx = rng.choice(adata.n_obs, size=n_sample, replace=False)

        calc_X  = adata.X[idx, :]
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

    # ── Save HVG stats to .uns BEFORE deleting calc_adata ──
    stat_cols = [c for c in calc_adata.var.columns
                 if c in ("highly_variable", "means",
                          "dispersions", "dispersions_norm",
                          "variances", "variances_norm")]
    hvg_stats = calc_adata.var[stat_cols].copy()
    adata.uns["hvg_stats"] = hvg_stats.to_dict("index")
    logger.info(f"  HVG stats saved to .uns['hvg_stats']")

    hvg_names = set(calc_adata.var_names[calc_adata.var["highly_variable"]])
    del calc_adata
    force_gc(logger)

    # ── Phase 8 Ghost Rescue ───────────────────────────────────────────────
    ghosts = [g for g in cc_in_matrix if g not in hvg_names]
    if ghosts:
        logger.info(f"  👻 HVG Ghost Rescue: {len(ghosts)} cell cycle genes added")
        hvg_names.update(ghosts)

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
            logger.info(f"  ⚡ HVG Target Rescue: {len(target_rescue)} perturbation targets forced in")
            hvg_names.update(target_rescue)

    # ── Apply HVG mask via C-buffer reconstructor ───
    keep_mask = np.array([g in hvg_names for g in adata.var_names])
    n_kept    = keep_mask.sum()
    logger.info(f"  Subsetting from {adata.n_vars:,} → {n_kept:,} features")

    adata.uns["spore_core_features"] = [
        g for g in adata.var_names if g in hvg_names and g not in set(ghosts)]

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

def print_hvg_diagnostics(adata):
    """
    Prints a clean Pandas DataFrame summary of the HVG selection, ghost rescues,
    and distribution metrics for each gene category.
    """
    import pandas as pd

    if "hvg_stats" not in adata.uns:
        print("No HVG stats found. Run Phase 8 first.")
        return

    hvg_df = pd.DataFrame.from_dict(adata.uns["hvg_stats"], orient="index")
    
    # 1. Feature Counts
    total_pre_genes = len(hvg_df)
    hvg_genes = hvg_df["highly_variable"].sum()
    non_hvg = total_pre_genes - hvg_genes
    final_features = len(adata.var_names)
    rescued_count = final_features - hvg_genes

    df_counts = pd.DataFrame({
        "Metric": [
            "Starting Features (Phase 6)",
            "Non-Variable (Filtered)",
            "Highly Variable (Retained)",
            "Total Rescues (Targets + Cell Cycle)",
            "Final Feature Space"
        ],
        "Count": [
            f"{total_pre_genes:,}", f"{non_hvg:,}", f"{hvg_genes:,}", 
            f"{rescued_count:,}", f"{final_features:,}"
        ],
        "Percentage": [
            "100.0%", f"{(non_hvg/total_pre_genes)*100:.1f}%", f"{(hvg_genes/total_pre_genes)*100:.1f}%",
            f"{(rescued_count/total_pre_genes)*100:.2f}%", f"{(final_features/total_pre_genes)*100:.1f}%"
        ]
    })

    print("\n" + "="*70)
    print(" SPORE+ HVG SELECTION DIAGNOSTICS")
    print("="*70)
    print(df_counts.to_string(index=False))
    
    # 2. Distribution Metrics
    # Determine which variance column was used
    y_col, y_name = None, None
    if "variances_norm" in hvg_df.columns:
        y_col, y_name = "variances_norm", "Variance"
    elif "dispersions_norm" in hvg_df.columns:
        y_col, y_name = "dispersions_norm", "Dispersion"

    if y_col:
        hvg_df["Status"] = "Non-Variable"
        hvg_df.loc[hvg_df["highly_variable"], "Status"] = "Highly Variable"
        
        if "spore_core_features" in adata.uns:
            core = set(adata.uns["spore_core_features"])
            ghosts = [g for g in adata.var_names if g not in core and g in hvg_df.index]
            if ghosts:
                hvg_df.loc[ghosts, "Status"] = "Rescued Targets"
        
        dist_data = []
        for status in ["Non-Variable", "Highly Variable", "Rescued Targets"]:
            sub = hvg_df[hvg_df["Status"] == status]
            if len(sub) == 0: continue
            
            dist_data.append({
                "Category": status,
                "Mean Expr (Avg)": f"{sub['means'].mean():.3f}",
                "Mean Expr (Range)": f"[{sub['means'].min():.3f}, {sub['means'].max():.2f}]",
                f"Norm. {y_name} (Avg)": f"{sub[y_col].mean():.3f}",
                f"Norm. {y_name} (Range)": f"[{sub[y_col].min():.3f}, {sub[y_col].max():.2f}]"
            })
            
        if dist_data:
            print("-" * 70)
            print(" DISTRIBUTION METRICS")
            print("-" * 70)
            print(pd.DataFrame(dist_data).to_string(index=False))

    print("="*70 + "\n")
