"""
SPORE+ · src/phase13_chitin.py
────────────────────────────────
Phase 13: CHITIN Correction

CHITIN is now absorbed into the SPORE+ pipeline as Phase 13.
It runs once per cell line on the metacell outputs from Phase 12.

Each cell line gets its own ChitinModel fitted on its train split controls
and applied to its train/val/test splits independently. This is the critical
fix for the multi-cell-line problem: each cell line has its own:
  - Systematic variation vector V = O_pert - C_ctrl
  - KNN reference manifold
  - Pareto-calibrated (k, n_pcs, metric) parameters

The CHITIN source code (engine.py, diagnostics.py, plotting.py) is
UNCHANGED — this wrapper simply calls it in the correct multi-cell-line loop.

Outputs: {dataset}_{cell_line}_{split}_metacell_chitin.h5ad
"""

import gc
from pathlib import Path

from .utils import log_phase_header, snapshot, log_memory, force_gc


def run_phase13(all_meta_splits: dict, cfg: dict, logger):
    """
    Run CHITIN correction for each cell line.

    Parameters
    ----------
    all_meta_splits : dict
        Output from Phase 12:
        { cell_line_name: { 'train': adata, 'val': adata, 'test': adata } }
    cfg : dict
        Unified SPORE+ config.
    logger : Logger

    Returns
    -------
    dict
        { cell_line_name: { 'train': adata_delta, 'val': adata_delta,
                            'test': adata_delta, 'model': ChitinModel } }
    """
    log_phase_header(logger, 13, "CHITIN Correction")
    p13 = cfg.get("phase13_chitin", {})

    if not p13.get("enabled", True):
        logger.info("  Phase 13: DISABLED in config")
        return {}

    # Build a CHITIN-compatible sub-config from the SPORE+ config
    chitin_cfg = _build_chitin_cfg(cfg)

    # Import CHITIN engine (works from src/ directory)
    try:
        from .engine import ChitinModel
    except ImportError:
        try:
            from engine import ChitinModel
        except ImportError:
            logger.error(
                "  CHITIN engine not found. "
                "Ensure src/engine.py (CHITIN) is present.")
            return {}

    out_dir      = cfg["paths"]["_chitin_output"]
    dataset_name = cfg.get("dataset", {}).get("name", "dataset")
    all_chitin   = {}

    for cell_line, splits in all_meta_splits.items():
        logger.info(f"  ══ CHITIN for cell line: {cell_line} ══")

        train_adata = splits.get("train")
        if train_adata is None:
            logger.warning(f"  No train split for {cell_line} — skipping CHITIN")
            continue

        # ── Fit ChitinModel on this cell line's train controls ─────────────
        logger.info(f"  [{cell_line}] Fitting CHITIN model on train split...")
        try:
            model = ChitinModel()
            train_adata = model.fit(train_adata, chitin_cfg, logger)
        except Exception as e:
            logger.error(f"  [{cell_line}] CHITIN fit failed: {e}")
            continue

        if model.selected_params:
            logger.info(
                f"  [{cell_line}] ★ Auto-selected: "
                f"k={model.k}, n_pcs={model.n_pcs}, "
                f"metric={model.distance_metric}")
        else:
            logger.info(
                f"  [{cell_line}] Manual params: "
                f"k={model.k}, n_pcs={model.n_pcs}, "
                f"metric={model.distance_metric}")

        cell_line_results = {"model": model}

        # ── Transform all splits ───────────────────────────────────────────
        cl_tag = f"_{cell_line}" if cell_line != "single" else ""
        suffix = p13.get("output", {}).get("suffix", "_chitin")

        for split_key in ["train", "val", "test"]:
            split_adata = splits.get(split_key)
            if split_adata is None:
                continue

            logger.info(f"  [{cell_line}] Transforming {split_key} split...")
            try:
                delta_adata = model.transform(
                    split_adata, chitin_cfg, logger,
                    label=f"{cell_line.upper()} {split_key.title()}")
            except Exception as e:
                logger.error(
                    f"  [{cell_line}] CHITIN transform({split_key}) failed: {e}")
                continue

            cell_line_results[split_key] = delta_adata

            # Save CHITIN output
            fname = (f"{dataset_name}{cl_tag}_{split_key}"
                     f"_metacell{suffix}.h5ad")
            path  = out_dir / fname
            delta_adata.write_h5ad(path)
            snapshot(delta_adata, f"CHITIN {cell_line}/{split_key}", logger)
            logger.info(f"  ✓ Saved CHITIN output → {path}")

        all_chitin[cell_line] = cell_line_results
        del model
        force_gc(logger)
        log_memory(logger, f"after {cell_line} CHITIN")

    logger.info(f"  Phase 13 complete: CHITIN applied to {len(all_chitin)} cell line(s)")
    return all_chitin


def _build_chitin_cfg(cfg: dict) -> dict:
    """
    Translate SPORE+ config into the format expected by ChitinModel.
    ChitinModel was written for CHITIN's own YAML structure;
    this adapter ensures compatibility.
    """
    p13 = cfg.get("phase13_chitin", {})
    p12 = cfg.get("phase12_metacell", {})
    ds  = cfg.get("dataset", {})

    chitin_cfg = {
        "dataset": {
            "name":             ds.get("name", "dataset"),
            "perturbation_col": ds.get("perturbation_col", "gene"),
            "control_label":    ds.get("control_label", "non-targeting"),
        },
        "knn": {
            "n_pcs":           p13.get("knn", {}).get("n_pcs", 50),
            "k":               p13.get("knn", {}).get("k", 20),
            "distance_metric": p13.get("knn", {}).get("distance_metric", "euclidean"),
            "svd_solver":      p13.get("knn", {}).get("svd_solver", "randomized"),
        },
        "calibration": p13.get("calibration", {
            "auto_calibrate": True,
            "k_grid":    [1, 3, 5, 10, 15, 20, 30, 50],
            "n_pcs_grid": [5, 10, 15, 20, 30, 50],
            "metric_grid": ["euclidean", "cosine"],
            "min_stability_fraction": 0.50,
        }),
        "correction": {
            "mode":                   p13.get("correction_mode", "knn"),
            "pc_systematic_threshold": p13.get("pc_systematic_threshold", 0.30),
        },
        "diagnostics": p13.get("diagnostics", {
            "k_sweep": True,
            "k_sweep_range": [5, 10, 15, 20, 30, 50, 75, 100],
            "compute_systema_cosines": True,
            "pc_decomposition_report": True,
            "pc_top_genes": 10,
        }),
        "output": p13.get("output", {
            "suffix": "_chitin",
            "preserve_systema_centroids": True,
            "save_sweep_results": True,
        }),
        "paths": {
            "_output":  cfg["paths"]["_chitin_output"],
            "_figures": cfg["paths"]["_figures"],
            "_logs":    cfg["paths"]["_logs"],
        },
        "plotting": cfg.get("plotting", {
            "style": "dark",
            "dpi": 200,
            "save_figures": True,
            "figure_format": "png",
        }),
        "runtime": cfg.get("runtime", {"n_jobs": 8}),
    }
    return chitin_cfg


def generate_chitin_report(all_chitin: dict, cfg: dict, logger):
    """
    Generate a brief CHITIN summary for inclusion in the SPORE+ final report.
    Returns a dict with per-cell-line CHITIN metrics.
    """
    summary = {}
    for cell_line, results in all_chitin.items():
        model = results.get("model")
        if model is None:
            continue
        entry = {
            "k":              model.k,
            "n_pcs":          model.n_pcs,
            "metric":         model.distance_metric,
            "correction_mode": model.correction_mode,
            "auto_calibrated": model.selected_params is not None,
            "systematic_pcs": model.systematic_pc_indices,
        }
        # Try to get sweep metrics
        if model.sweep_results is not None and len(model.sweep_results) > 0:
            sel = model.selected_params
            if sel:
                df = model.sweep_results
                match = df[
                    (df["k"] == sel["k"]) &
                    (df["n_pcs"] == sel["n_pcs"]) &
                    (df["metric"] == sel["metric"])]
                if len(match) > 0:
                    row = match.iloc[0]
                    entry["rank_disruption"]  = float(row.get("rank_disruption", 0))
                    entry["disc_ratio"]       = float(row.get("disc_ratio", 0))
                    entry["signal_stability"] = float(row.get("signal_stability", 0))
        summary[cell_line] = entry
    return summary
