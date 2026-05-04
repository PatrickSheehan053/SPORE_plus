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
    zs        = cfg.get("phase7_splits", {}).get("zero_shot", {})
    pert_col  = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]

    if seed is None:
        seed = cfg.get("phase7_splits", {}).get("random_seed", 42)
    rng = np.random.default_rng(seed)

    deg_counts = _compute_mean_shift(adata, cfg, logger)
    train_labels, test_labels = _stratified_split(
        deg_counts,
        zs.get("train_test_ratio", 0.90),
        zs.get("stratify_bins", 4),
        rng)

    val_ratio = zs.get("validation_ratio", 0.20)
    n_val     = max(1, int(len(train_labels) * val_ratio))
    rng.shuffle(train_labels)
    val_labels   = train_labels[:n_val]
    train_labels = train_labels[n_val:]

    logger.info(
        f"  Split: {len(train_labels)} train / {len(val_labels)} val / "
        f"{len(test_labels)} test perturbations")

    pert_values = adata.obs[pert_col].values
    train_set   = set(train_labels)
    val_set     = set(val_labels)
    test_set    = set(test_labels)

    train_mask = np.array([p in train_set or p == ctrl_label for p in pert_values])
    val_mask   = np.array([p in val_set   or p == ctrl_label for p in pert_values])
    test_mask  = np.array([p in test_set                     for p in pert_values])

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

    # Sanity check: val should be larger than test (controls in val, not test)
    if val_ad.n_obs < test_ad.n_obs:
        logger.warning(
            f"  ⚠ val ({val_ad.n_obs:,}) < test ({test_ad.n_obs:,}). "
            f"Increase validation_ratio (currently "
            f"{zs.get('validation_ratio', 0.20)}) or the test perturbations "
            f"have more cells than val. Downstream tools expect val ≥ test.")

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
