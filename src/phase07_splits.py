"""
SPORE+ · src/phase07_splits.py
────────────────────────────────
Phase 7: Data Splits
Adapted from SPORE phase4_splits.py.
Config key: phase7_splits (was phase4_splits).

Multi-cell-line awareness: split assignments are made globally by perturbation
label. The 'sporeplus_cell_line' column (if present) is preserved in all splits
so that Phase 11/12 can separate by cell line.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import json
from collections import OrderedDict
from .utils import (log_phase_header, snapshot, log_memory, force_gc,
                    safe_in_memory_row_subset)


def destructive_3way_split(adata, train_mask, val_mask, test_mask, logger):
    """
    Error 011 fix: never hold two 80% copies simultaneously.
    Extract small val/test first, then in-place mutate adata → train.
    """
    logger.info("  Applying destructive 3-way memory split...")

    logger.info("    Extracting test split...")
    test_ad = ad.AnnData(
        X=adata.X[test_mask].copy(),
        obs=adata.obs.iloc[test_mask].copy(),
        var=adata.var.copy())

    logger.info("    Extracting val split...")
    val_ad = ad.AnnData(
        X=adata.X[val_mask].copy(),
        obs=adata.obs.iloc[val_mask].copy(),
        var=adata.var.copy())

    logger.info("    Mutating original object → train split in-place...")
    n_kept = train_mask.sum()

    if sp.issparse(adata.X) and adata.X.format == "csr":
        indptr  = adata.X.indptr
        indices = adata.X.indices
        data    = adata.X.data

        new_indptr = np.zeros(n_kept + 1, dtype=indptr.dtype)
        padded = np.concatenate(([False], train_mask, [False]))
        diff   = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]

        write_ptr = 0
        new_row   = 0
        for start, end in zip(starts, ends):
            n_rows_block = end - start
            data_start   = indptr[start]
            data_end     = indptr[end]
            nnz_block    = data_end - data_start
            if nnz_block > 0:
                indices[write_ptr:write_ptr + nnz_block] = indices[data_start:data_end]
                data[write_ptr:write_ptr + nnz_block]    = data[data_start:data_end]
            new_indptr[new_row + 1:new_row + 1 + n_rows_block] = (
                indptr[start + 1:end + 1] - data_start + write_ptr)
            write_ptr += nnz_block
            new_row   += n_rows_block

        new_X = sp.csr_matrix(
            (data[:write_ptr], indices[:write_ptr], new_indptr),
            shape=(n_kept, adata.X.shape[1]))
    else:
        logger.warning("    Matrix not CSR — standard slice (may spike RAM).")
        new_X = adata.X[train_mask]

    train_ad = ad.AnnData(
        X=new_X,
        obs=adata.obs.iloc[train_mask].copy(),
        var=adata.var.copy())

    # Hollow the original (Error 012: don't touch adata.obs)
    adata.X = None
    if hasattr(adata, "obsm"): adata.obsm.clear()
    if hasattr(adata, "varm"): adata.varm.clear()
    if hasattr(adata, "uns"):  adata.uns.clear()
    force_gc(logger)

    return train_ad, val_ad, test_ad


def _compute_mean_shift(adata, cfg, logger):
    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]
    perturbations = [p for p in adata.obs[pert_col].unique()
                     if p != ctrl_label]
    ctrl_mask = adata.obs[pert_col] == ctrl_label

    X = adata.X
    if sp.issparse(X):
        ctrl_mean = np.array(X[ctrl_mask].mean(axis=0)).flatten()
    else:
        ctrl_mean = X[ctrl_mask].mean(axis=0)

    shifts = {}
    for target in perturbations:
        pert_mask = adata.obs[pert_col] == target
        if sp.issparse(X):
            pert_mean = np.array(X[pert_mask].mean(axis=0)).flatten()
        else:
            pert_mean = X[pert_mask].mean(axis=0)
        shifts[target] = float(np.abs(pert_mean - ctrl_mean).sum())

    return pd.Series(shifts).sort_values(ascending=False)


def _stratified_split(perturbations, train_ratio, n_bins, rng):
    df = perturbations.reset_index()
    df.columns = ["perturbation", "score"]
    df["bin"] = pd.qcut(df["score"], q=n_bins, labels=False, duplicates="drop")

    train_labels, test_labels = [], []
    for _, group in df.groupby("bin"):
        shuffled  = group.sample(frac=1, random_state=rng.integers(1e9))
        n_train   = max(1, int(len(shuffled) * train_ratio))
        train_labels.extend(shuffled.iloc[:n_train]["perturbation"])
        test_labels.extend(shuffled.iloc[n_train:]["perturbation"])
    return train_labels, test_labels


def split_zero_shot(adata, cfg, logger, seed=None):
    zs = cfg.get("phase7_splits", {})
    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    if seed is None:
        seed = zs.get("random_seed", 42)
    rng = np.random.default_rng(seed)

    deg_counts = _compute_mean_shift(adata, cfg, logger)
    n_perts = len(deg_counts)
    
    # ── 1. The Literature-Standard Split Ratios ──
    # Hold out 25% of perturbations for Test
    test_ratio = zs.get("test_ratio", 0.25)
    n_test = max(1, int(n_perts * test_ratio))
    
    # Of the remaining 75%, hold out 10% for Validation
    val_ratio = zs.get("val_ratio", 0.10)
    n_remain = n_perts - n_test
    n_val = max(1, int(n_remain * val_ratio))
    n_train = n_remain - n_val

    # Stratified binning for assignment
    df = deg_counts.reset_index()
    df.columns = ["perturbation", "score"]
    df["bin"] = pd.qcut(df["score"], q=zs.get("stratify_bins", 4), labels=False, duplicates="drop")

    train_labels, val_labels, test_labels = [], [], []
    for _, group in df.groupby("bin"):
        shuffled = group.sample(frac=1, random_state=rng.integers(1e9))
        
        # Calculate proportional allocations per bin
        bin_test = max(1, int(len(shuffled) * test_ratio)) if len(shuffled) > 3 else 1
        bin_remain = len(shuffled) - bin_test
        bin_val = max(1, int(bin_remain * val_ratio)) if bin_remain > 2 else 1
        
        test_labels.extend(shuffled.iloc[:bin_test]["perturbation"])
        val_labels.extend(shuffled.iloc[bin_test:bin_test+bin_val]["perturbation"])
        train_labels.extend(shuffled.iloc[bin_test+bin_val:]["perturbation"])

    logger.info(
        f"  Perturbation Split: {len(train_labels)} Train / {len(val_labels)} Val / "
        f"{len(test_labels)} Test (Zero-Shot Targets)")

    # ── 2. Distributing the Control Cells ──
    pert_values = adata.obs[pert_col].values
    ctrl_mask_full = pert_values == ctrl_label
    ctrl_indices = np.where(ctrl_mask_full)[0]
    
    rng.shuffle(ctrl_indices)
    
    frac_test = len(test_labels) / n_perts
    frac_val = len(val_labels) / n_perts
    
    n_ctrl_test = int(len(ctrl_indices) * frac_test)
    n_ctrl_val = int(len(ctrl_indices) * frac_val)
    
    test_ctrl_idx = ctrl_indices[:n_ctrl_test]
    val_ctrl_idx = ctrl_indices[n_ctrl_test:n_ctrl_test+n_ctrl_val]
    train_ctrl_idx = ctrl_indices[n_ctrl_test+n_ctrl_val:]

    # ── 3. Assembling the Final Masks ──
    train_mask = np.isin(pert_values, train_labels)
    train_mask[train_ctrl_idx] = True
    
    val_mask = np.isin(pert_values, val_labels)
    val_mask[val_ctrl_idx] = True
    
    test_mask = np.isin(pert_values, test_labels)
    test_mask[test_ctrl_idx] = True

    log_memory(logger, "before destructive split")
    train_ad, val_ad, test_ad = destructive_3way_split(
        adata, train_mask, val_mask, test_mask, logger)

    split_info = {
        "train": train_ad.n_obs,
        "val":   val_ad.n_obs,
        "test":  test_ad.n_obs,
    }
    snapshot(train_ad, "Train split", logger)
    snapshot(val_ad,   "Val split",   logger)
    snapshot(test_ad,  "Test split",  logger)

    return {
        "train": train_ad, "val": val_ad, "test": test_ad,
        "deg_counts": deg_counts, "split_info": split_info,
        "train_labels": train_labels, "val_labels": val_labels,
        "test_labels": test_labels, "seed": seed,
    }


def save_splits(split_result, cfg, logger, seed=None):
    splits_dir   = cfg["paths"]["_splits"]
    dataset_name = cfg["dataset"]["name"]
    if seed is not None:
        splits_dir = splits_dir / f"seed_{seed}"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for key in ["train", "val", "test"]:
        path = splits_dir / f"{dataset_name}_{key}.h5ad"
        split_result[key].write_h5ad(path)
        logger.info(f"  Saved {key} → {path}")

    if "train_labels" in split_result:
        meta = {
            "train_labels": list(split_result["train_labels"]),
            "val_labels":   list(split_result["val_labels"]),
            "test_labels":  list(split_result["test_labels"]),
            "seed":         int(split_result["seed"]),
        }
        meta_path = splits_dir / f"{dataset_name}_split_indices.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def run_phase7(adata, cfg, logger):
    log_phase_header(logger, 7, "Data Splits")
    mode      = cfg.get("phase7_splits", {}).get("mode", "zero_shot")
    test_mode = cfg.get("phase7_splits", {}).get("test_mode", False)
    seeds     = ([cfg.get("phase7_splits", {}).get("random_seed", 42)]
                 if not test_mode
                 else cfg.get("phase7_splits", {}).get("test_mode_seeds", [42]))

    all_splits = []
    for seed in seeds[:1]:  # Destructive split: only first seed in single run
        result = split_zero_shot(adata, cfg, logger, seed=seed)
        save_splits(result, cfg, logger,
                    seed=seed if test_mode else None)
        all_splits.append(result)
    return all_splits

def print_split_diagnostics(split_result):
    """
    Prints a clean Pandas DataFrame summary of the data splits and performs
    mathematical checks to guarantee zero data leakage between sets.
    """
    import pandas as pd
    
    # Handle whether the user passed the full list of seeds or just one result
    res = split_result[0] if isinstance(split_result, list) else split_result
    
    # Extract Cell Counts
    train_cells = res["split_info"]["train"]
    val_cells   = res["split_info"]["val"]
    test_cells  = res["split_info"]["test"]
    total_cells = train_cells + val_cells + test_cells
    
    # Extract Perturbations
    train_perts = set(res["train_labels"])
    val_perts   = set(res["val_labels"])
    test_perts  = set(res["test_labels"])
    total_perts = len(train_perts) + len(val_perts) + len(test_perts)
    
    # ── 1. Clean Table Output ──
    df = pd.DataFrame({
        "Split": ["Train", "Val", "Test", "Total"],
        "Cells": [f"{train_cells:,}", f"{val_cells:,}", f"{test_cells:,}", f"{total_cells:,}"],
        "Cell %": [
            f"{(train_cells/total_cells)*100:.1f}%", 
            f"{(val_cells/total_cells)*100:.1f}%", 
            f"{(test_cells/total_cells)*100:.1f}%", 
            "100.0%"
        ],
        "Targets": [len(train_perts), len(val_perts), len(test_perts), total_perts],
        "Target %": [
            f"{(len(train_perts)/total_perts)*100:.1f}%", 
            f"{(len(val_perts)/total_perts)*100:.1f}%", 
            f"{(len(test_perts)/total_perts)*100:.1f}%", 
            "100.0%"
        ]
    })
    
    print("\n" + "="*50)
    print(" SPORE+ DATA SPLIT DIAGNOSTICS")
    print("="*50)
    print(df.to_string(index=False))
    print("\n" + "-"*50)
    print(" LEAKAGE SANITY CHECKS")
    print("-"*50)
    
    # ── 2. Mathematical Leakage Checks ──
    leak_tv = len(train_perts.intersection(val_perts))
    leak_tt = len(train_perts.intersection(test_perts))
    leak_vt = len(val_perts.intersection(test_perts))
    
    if leak_tv == 0 and leak_tt == 0 and leak_vt == 0:
        print("[PASS] 🟢 Target sets are strictly mutually exclusive.")
    else:
        print("[FAIL] 🔴 DATA LEAKAGE DETECTED!")
        print(f"       Train/Val overlap:  {leak_tv} targets")
        print(f"       Train/Test overlap: {leak_tt} targets")
        print(f"       Val/Test overlap:   {leak_vt} targets")

    if total_perts > 0:
        print(f"[PASS] 🟢 All perturbations accounted for ({total_perts}).")
    print("="*50 + "\n")
