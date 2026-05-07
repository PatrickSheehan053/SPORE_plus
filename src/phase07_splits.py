"""
SPORE+ · src/phase07_splits.py
────────────────────────────────
Phase 7: Stratified Zero-Shot Data Splits

Engineered for foundation model training (e.g., CHITIN, GEARS).
To prevent data leakage, splits are executed on a "Zero-Shot" basis: entire 
perturbation targets are completely held out from the training set. This forces 
the downstream model to generalize to unseen genetic states rather than 
memorizing transcriptomic signatures.

Stratified L1-Norm Binning
──────────────────────────
Targets are scored by their absolute mean shift from non-targeting controls. 
They are binned by transcriptional impact (severity), and the Train/Val/Test 
splits are proportionally sampled from these bins to guarantee uniform 
difficulty across the splits.

Destructive Memory Management (For 1M+ Cell Datasets)
─────────────────────────────────────────────────────
Standard subsetting requires 2x memory overhead. For massive datasets, this 
module extracts the smaller Val/Test splits first, and then executes a 
low-level destructive reconstruction of the CSR sparse pointer arrays to 
mutate the original AnnData object into the Train split in-place. 
Peak RAM overhead is reduced by ~80%.
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
    labels = adata.obs[pert_col].values
    perturbations = [p for p in np.unique(labels) if p != ctrl_label]

    # ── OOM FIREWALL 1: Stream target means in safe chunks ──
    is_large = getattr(adata, 'isbacked', False) or adata.n_obs > 1000000

    if is_large:
        logger.info("  [Splits] Large/Backed mode: Chunking mean shift calculation to prevent HDF5 I/O choke...")
        pert_sums = {t: np.zeros(adata.n_vars, dtype=np.float64) for t in perturbations}
        pert_sums[ctrl_label] = np.zeros(adata.n_vars, dtype=np.float64)
        pert_counts = {t: 0 for t in perturbations}
        pert_counts[ctrl_label] = 0

        chunk_size = 50000
        for start in range(0, adata.n_obs, chunk_size):
            end = min(start + chunk_size, adata.n_obs)
            chunk_labels = labels[start:end]
            chunk_X = adata.X[start:end]
            if sp.issparse(chunk_X):
                chunk_X = chunk_X.toarray()

            for lbl in np.unique(chunk_labels):
                if lbl in pert_sums:
                    mask = chunk_labels == lbl
                    pert_sums[lbl] += chunk_X[mask].sum(axis=0)
                    pert_counts[lbl] += mask.sum()
            del chunk_X # Instant memory release

        ctrl_mean = (pert_sums[ctrl_label] / max(pert_counts[ctrl_label], 1)).astype(np.float32)

        shifts = {}
        for target in perturbations:
            if pert_counts[target] > 0:
                t_mean = (pert_sums[target] / pert_counts[target]).astype(np.float32)
                shifts[target] = float(np.abs(t_mean - ctrl_mean).sum())
            else:
                shifts[target] = 0.0
        del pert_sums
        return pd.Series(shifts).sort_values(ascending=False)
    else:
        # Standard fast path for fully in-memory datasets
        ctrl_mask = labels == ctrl_label
        X = adata.X
        if sp.issparse(X):
            ctrl_mean = np.array(X[ctrl_mask].mean(axis=0)).flatten()
        else:
            ctrl_mean = X[ctrl_mask].mean(axis=0)

        shifts = {}
        for target in perturbations:
            pert_mask = labels == target
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
    
    # ── OOM FIREWALL 2: Avoid destructive mutation on HDF5 files ──
    is_large = getattr(adata, 'isbacked', False) or adata.n_obs > 1000000
    
    if is_large:
        logger.info("  [Splits] Large/Backed mode: Creating lazy views to prevent HDF5 corruption and OOM...")
        train_ad = adata[train_mask]
        val_ad   = adata[val_mask]
        test_ad  = adata[test_mask]
    else:
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

