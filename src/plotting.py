"""
SPORE+ · src/plotting.py  (v2)
────────────────────────────────
Publication-quality plots for the SPORE+ pipeline.

Design philosophy
─────────────────
• White / seaborn "paper" background — no dark navy blocks
• Information density: every figure encodes ≥2 variables simultaneously
• rocket / inferno / mako colourmaps for continuous data
• Layered geometry: KDE fill + contour lines + marginal density where applicable
• Compact grid layout — figures sized to fit a laptop screen without scrolling
• Bold panel-letter labels (A, B, C …) for multi-panel figures
• Text outputs replace trivial single-number plots entirely

Colour conventions
──────────────────
  Perturbed / foreground:  #d45f00  (warm amber-orange)
  Control / reference:     #2c7fb8  (steel blue)
  Accent:                  #e31a1c  (bright red)
  Accent 2:                #6a3d9a  (purple)
  Neutral:                 #444444
  Background:              #ffffff  (always white)
  Grid / spines:           #dddddd
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")

_THEME = {
    "bg":      "#ffffff",
    "grid":    "#dddddd",
    "spine":   "#bbbbbb",
    "pert":    "#d45f00",
    "ctrl":    "#2c7fb8",
    "accent":  "#e31a1c",
    "accent2": "#6a3d9a",
    "neutral": "#444444",
    "muted":   "#888888",
    "good":    "#2ca25f",
    "warn":    "#e31a1c",
}


def _apply_theme():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    mpl.rcParams.update({
        "figure.facecolor":  _THEME["bg"],
        "axes.facecolor":    _THEME["bg"],
        "axes.edgecolor":    _THEME["spine"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        _THEME["grid"],
        "grid.linewidth":    0.6,
        "axes.labelcolor":   _THEME["neutral"],
        "xtick.color":       _THEME["neutral"],
        "ytick.color":       _THEME["neutral"],
        "text.color":        _THEME["neutral"],
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "legend.frameon":    False,
        "legend.fontsize":   10,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
        "figure.dpi":        120,
    })


_apply_theme()


def _label(ax, letter):
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="left",
            color=_THEME["neutral"])


def _save(fig, cfg, stem):
    figs = cfg.get("paths", {}).get("_figures")
    if figs is None:
        return None
    path = Path(figs) / f"{stem}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=_THEME["bg"])
    return path


def _show(fig):
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 0 · Sparsity overview
# ─────────────────────────────────────────────────────────────────────────────

def plot_sparsity_overview(adata, cfg):
    _apply_theme()
    import scipy.sparse as sp
    X = adata.X
    cpg = np.asarray((X > 0).sum(axis=0)).ravel() if sp.issparse(X) else (X > 0).sum(axis=0)
    gpc = np.asarray((X > 0).sum(axis=1)).ravel() if sp.issparse(X) else (X > 0).sum(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(cpg, bins=80, color=_THEME["ctrl"], edgecolor="white", lw=0.3, alpha=0.85)
    ax1.set_xscale("log")
    ax1.set_xlabel("Cells expressing gene (log)")
    ax1.set_ylabel("Genes")
    ax1.set_title("A  Gene detection frequency", loc="left", fontweight="bold")
    _label(ax1, "A")

    sorted_g = np.sort(gpc)
    ecdf = np.arange(1, len(sorted_g) + 1) / len(sorted_g)
    ax2.plot(sorted_g, ecdf, color=_THEME["pert"], lw=2)
    ax2.axvline(np.median(gpc), color=_THEME["accent"], ls="--", lw=1.2,
                label=f"Median: {int(np.median(gpc)):,}")
    ax2.set_xlabel("Genes per cell")
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_title("B  Genes per cell (ECDF)", loc="left", fontweight="bold")
    ax2.legend()
    _label(ax2, "B")

    fig.suptitle(f"Sparsity · {adata.n_obs:,} cells × {adata.n_vars:,} genes",
                 fontsize=13, fontweight="bold", y=1.01)
    path = _save(fig, cfg, "p00_sparsity")
    _show(fig)
    return fig, path


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 · QC
# ─────────────────────────────────────────────────────────────────────────────

def plot_qc_violin(adata, cfg):
    _apply_theme()
    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]
    metrics    = [m for m in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
                  if m in adata.obs.columns]
    if not metrics:
        return None, None

    obs = adata.obs.copy()
    obs["_g"] = obs[pert_col].apply(lambda x: "Control" if str(x) == ctrl_label else "Perturbed")

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    labels   = ["Genes detected", "Total UMI counts", "MT fraction (%)"]
    palette  = {"Control": _THEME["ctrl"], "Perturbed": _THEME["pert"]}

    for i, (m, lbl) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        sns.violinplot(data=obs, x="_g", y=m, ax=ax, palette=palette,
                       inner="quartile", cut=0, linewidth=1.2, alpha=0.85,
                       order=["Control", "Perturbed"])
        ax.set_xlabel("")
        ax.set_ylabel(lbl)
        ax.set_title(f"{chr(65+i)}  {lbl}", loc="left", fontweight="bold")
        _label(ax, chr(65+i))

    fig.suptitle("QC metrics — pre-filter", fontsize=13, fontweight="bold", y=1.02)
    path = _save(fig, cfg, "p02_qc_violins")
    _show(fig)
    return fig, path


def plot_mt_scatter(adata, cfg):
    _apply_theme()
    obs = adata.obs
    if not all(c in obs.columns for c in ["total_counts", "pct_counts_mt", "n_genes_by_counts"]):
        return None, None
    s = obs.sample(min(50_000, len(obs)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(s["total_counts"], s["pct_counts_mt"],
                    c=s["n_genes_by_counts"], cmap="mako", s=2, alpha=0.5, rasterized=True)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Genes detected")
    ax.set_xscale("log")
    ax.set_xlabel("Total UMI (log)")
    ax.set_ylabel("MT fraction (%)")
    ax.set_title("A  MT% vs total counts", loc="left", fontweight="bold")
    _label(ax, "A")
    path = _save(fig, cfg, "p02_mt_scatter")
    _show(fig)
    return fig, path


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 5 · Escaper summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_escaper_summary(escaper_stats, pert_sizes, cfg):
    _apply_theme()
    if escaper_stats is None or len(escaper_stats) == 0:
        return None, None

    df = escaper_stats.copy()
    pert_col = "perturbation" if "perturbation" in df.columns else df.columns[0]
    df["esc_frac"] = df["n_escaped"] / (df["n_total"] + 1e-9)
    df = df.sort_values("esc_frac", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(4, min(len(df) * 0.22, 9))))

    show = df.head(30)
    colors = [_THEME["accent"] if f > 0.5 else _THEME["pert"] for f in show["esc_frac"]]
    ax1.barh(range(len(show)), show["esc_frac"].values, color=colors,
             edgecolor="white", lw=0.3, height=0.7)
    ticks = show[pert_col].tolist() if pert_col in show.columns else [""] * len(show)
    ax1.set_yticks(range(len(show)))
    ax1.set_yticklabels(ticks, fontsize=8)
    ax1.invert_yaxis()
    ax1.axvline(0.5, color=_THEME["neutral"], ls="--", lw=0.8, alpha=0.6)
    ax1.set_xlabel("Escaper fraction")
    ax1.set_title("A  Escaper rate by perturbation (top 30)", loc="left", fontweight="bold")
    _label(ax1, "A")

    ax2.hist(df["esc_frac"], bins=40, color=_THEME["ctrl"],
             edgecolor="white", lw=0.3, alpha=0.85)
    ax2.axvline(df["esc_frac"].median(), color=_THEME["accent"], ls="--", lw=1.2,
                label=f"Median: {df['esc_frac'].median():.3f}")
    ax2.set_xlabel("Escaper fraction")
    ax2.set_ylabel("Perturbations")
    ax2.legend()
    ax2.set_title("B  Escaper fraction distribution", loc="left", fontweight="bold")
    _label(ax2, "B")

    fig.suptitle("Phase 5 · Escaper filtering", fontsize=13, fontweight="bold", y=1.01)
    path = _save(fig, cfg, "p05_escaper_summary")
    _show(fig)
    return fig, path


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 6 · Gene expression histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_gene_expression_hist(adata, cfg):
    _apply_theme()
    import scipy.sparse as sp
    X = adata.X
    gene_means = np.asarray(X.mean(axis=0)).ravel() if sp.issparse(X) else X.mean(axis=0)
    n_expr     = np.asarray((X > 0).sum(axis=0)).ravel() if sp.issparse(X) else (X > 0).sum(axis=0)
    min_cells  = cfg.get("preprocessing", {}).get("min_cells_per_gene", 10)
    keep = n_expr >= min_cells

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(np.log1p(gene_means[keep]),  bins=80, color=_THEME["ctrl"],
            edgecolor="white", lw=0.2, alpha=0.75, label=f"Kept ({keep.sum():,})")
    ax.hist(np.log1p(gene_means[~keep]), bins=80, color=_THEME["accent"],
            edgecolor="white", lw=0.2, alpha=0.65, label=f"Removed ({(~keep).sum():,})")
    ax.set_xlabel("log(1 + mean expression)")
    ax.set_ylabel("Genes")
    ax.legend()
    ax.set_title("A  Gene expression distribution — pre gene triage",
                 loc="left", fontweight="bold")
    _label(ax, "A")
    path = _save(fig, cfg, "p06_gene_expression")
    _show(fig)
    return fig, path


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 7 · Split summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_split_summary(split_result, cfg):
    _apply_theme()
    if split_result is None:
        return None, None
    pert_col = cfg["dataset"]["perturbation_col"]
    splits   = {k: split_result[k] for k in ["train", "val", "test"] if k in split_result}
    sizes    = {k: v.n_obs for k, v in splits.items() if v is not None}
    if not sizes:
        return None, None
    total = sum(sizes.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    names  = list(sizes.keys())
    counts = list(sizes.values())
    colors = [_THEME["ctrl"], _THEME["pert"], _THEME["accent2"]]
    bars   = ax1.barh(names, counts, color=colors[:len(names)],
                      edgecolor="white", lw=0.5, height=0.5)
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_width() + total * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{c:,}  ({c/total*100:.1f}%)", va="center", fontsize=10)
    ax1.set_xlim(0, total * 1.3)
    ax1.set_xlabel("Cells")
    ax1.set_title("A  Cell count by split", loc="left", fontweight="bold")
    _label(ax1, "A")

    for (name, adata_s), color in zip(splits.items(), colors):
        if adata_s is None or pert_col not in adata_s.obs.columns:
            continue
        pc = adata_s.obs[pert_col].value_counts().values.astype(float)
        if len(pc) > 1:
            sns.kdeplot(pc, ax=ax2, label=name.capitalize(),
                        color=color, fill=True, alpha=0.3, lw=1.8)
    ax2.set_xlabel("Cells per perturbation")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.set_title("B  Perturbation size distributions", loc="left", fontweight="bold")
    _label(ax2, "B")

    fig.suptitle("Data split summary", fontsize=13, fontweight="bold", y=1.02)
    path = _save(fig, cfg, "p07_split_summary")
    _show(fig)
    return fig, path


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 8 · HVG mean-variance
# ─────────────────────────────────────────────────────────────────────────────

def plot_hvg_variance(adata, cfg):
    _apply_theme()
    if "highly_variable" not in adata.var.columns:
        return None, None
    var = adata.var.copy()
    if "means" not in var.columns:
        var["means"] = var.get("mean_counts", np.nan)
    if "dispersions_norm" not in var.columns:
        var["dispersions_norm"] = var.get("residual_variances", np.nan)
    if var[["means", "dispersions_norm"]].isnull().all().all():
        return None, None

    hvg = var[var["highly_variable"]].dropna(subset=["means", "dispersions_norm"])
    non = var[~var["highly_variable"]].dropna(subset=["means", "dispersions_norm"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(non["means"], non["dispersions_norm"], s=4, color=_THEME["muted"],
               alpha=0.3, rasterized=True, label=f"Non-HVG ({len(non):,})")
    hvg_s = hvg.sort_values("dispersions_norm")
    rank  = np.arange(len(hvg_s)) / max(len(hvg_s) - 1, 1)
    sc    = ax.scatter(hvg_s["means"], hvg_s["dispersions_norm"], c=rank,
                       cmap="rocket", s=14, alpha=0.9, zorder=3, rasterized=True)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="HVG dispersion rank")
    ax.set_xscale("log")
    ax.set_xlabel("Mean expression (log)")
    ax.set_ylabel("Normalised dispersion")
    ax.set_title(f"A  HVG selection — {len(hvg):,} / {len(var):,} genes",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper left")
    _label(ax, "A")
    path = _save(fig, cfg, "p08_hvg_variance")
    _show(fig)
    return fig, path


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 10 · UMAP (layered bivariate)
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap(adata, cfg, color_by=None):
    _apply_theme()
    if "X_umap" not in adata.obsm:
        return None, None
    pert_col   = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]
    umap = adata.obsm["X_umap"]
    obs  = adata.obs.copy()
    obs["_u1"]       = umap[:, 0]
    obs["_u2"]       = umap[:, 1]
    obs["_is_ctrl"]  = obs[pert_col].astype(str) == ctrl_label
    rng = np.random.default_rng(42)
    pert_df = obs[~obs["_is_ctrl"]]
    ctrl_df = obs[obs["_is_ctrl"]]
    if len(pert_df) > 80_000:
        pert_df = pert_df.iloc[rng.choice(len(pert_df), 80_000, replace=False)]
    if len(ctrl_df) > 10_000:
        ctrl_df = ctrl_df.iloc[rng.choice(len(ctrl_df), 10_000, replace=False)]

    fig = plt.figure(figsize=(8, 7))
    outer = gridspec.GridSpec(1, 1, left=0.1, right=0.88, top=0.88, bottom=0.1)
    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0],
                                              width_ratios=[5, 1], height_ratios=[1, 5],
                                              wspace=0.05, hspace=0.05)
    ax_m = fig.add_subplot(inner[1, 0])
    ax_t = fig.add_subplot(inner[0, 0], sharex=ax_m)
    ax_r = fig.add_subplot(inner[1, 1], sharey=ax_m)

    if len(pert_df) > 200:
        sns.kdeplot(data=pert_df, x="_u1", y="_u2", ax=ax_m,
                    cmap="mako", fill=True, thresh=0.02, levels=15, alpha=0.85)
        sns.kdeplot(data=pert_df, x="_u1", y="_u2", ax=ax_m,
                    color="#111111", linewidths=0.4, thresh=0.05, levels=10, alpha=0.45)
    ax_m.scatter(ctrl_df["_u1"], ctrl_df["_u2"], c=_THEME["ctrl"],
                 s=6, alpha=0.75, zorder=4, rasterized=True)

    sns.kdeplot(data=pert_df, x="_u1", ax=ax_t, color=_THEME["pert"],
                fill=True, alpha=0.5, lw=1.5, label="Perturbed")
    sns.kdeplot(data=ctrl_df, x="_u1", ax=ax_t, color=_THEME["ctrl"],
                fill=True, alpha=0.6, lw=1.5, label="Control")
    sns.kdeplot(data=pert_df, y="_u2", ax=ax_r, color=_THEME["pert"],
                fill=True, alpha=0.5, lw=1.5)
    sns.kdeplot(data=ctrl_df, y="_u2", ax=ax_r, color=_THEME["ctrl"],
                fill=True, alpha=0.6, lw=1.5)
    ax_t.set_axis_off()
    ax_r.set_axis_off()
    ax_m.set_xlabel("UMAP 1")
    ax_m.set_ylabel("UMAP 2")
    handles = [mpatches.Patch(color="#2d1e6b", alpha=0.8, label=f"Perturbed ({len(pert_df):,})"),
               mpatches.Patch(color=_THEME["ctrl"], label=f"Control ({len(ctrl_df):,})")]
    ax_m.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(f"UMAP · {adata.n_obs:,} cells",
                 fontsize=13, fontweight="bold", y=0.97)
    stem = f"p10_umap_{color_by}" if color_by else "p10_umap"
    path = _save(fig, cfg, stem)
    _show(fig)
    return fig, path


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 11 · Cell line PCA (layered bivariate with marginals)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cell_line_pca(adata, cell_line_labels, n_cell_lines, cfg):
    _apply_theme()
    pca_key = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
    if pca_key not in adata.obsm:
        return None, None
    pca = adata.obsm[pca_key]
    obs = adata.obs.copy()
    obs["_pc1"]       = pca[:, 0]
    obs["_pc2"]       = pca[:, 1]
    pert_col          = cfg["dataset"]["perturbation_col"]
    ctrl_label        = cfg["dataset"]["control_label"]
    obs["_is_ctrl"]   = obs[pert_col].astype(str) == ctrl_label
    obs["_cell_line"] = (cell_line_labels if cell_line_labels is not None
                         else np.full(len(obs), "single"))

    rng = np.random.default_rng(42)
    if len(obs) > 120_000:
        obs = obs.iloc[rng.choice(len(obs), 120_000, replace=False)]
    ctrl_s = obs[obs["_is_ctrl"]]
    pert_s = obs[~obs["_is_ctrl"]]
    multi  = n_cell_lines > 1

    fig = plt.figure(figsize=(8.5, 7.5))
    outer = gridspec.GridSpec(1, 1, left=0.1, right=0.88, top=0.88, bottom=0.1)
    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0],
                                              width_ratios=[5, 1], height_ratios=[1, 5],
                                              wspace=0.05, hspace=0.05)
    ax_m = fig.add_subplot(inner[1, 0])
    ax_t = fig.add_subplot(inner[0, 0], sharex=ax_m)
    ax_r = fig.add_subplot(inner[1, 1], sharey=ax_m)

    line_names  = obs["_cell_line"].unique().tolist() if multi else ["single"]
    line_colors = sns.color_palette("tab10", n_colors=max(n_cell_lines, 2))

    if multi:
        for lname, lcolor in zip(line_names, line_colors):
            sub = obs[obs["_cell_line"] == lname]
            if len(sub) < 100:
                continue
            sns.kdeplot(data=sub, x="_pc1", y="_pc2", ax=ax_m, color=lcolor,
                        fill=True, thresh=0.04, levels=12, alpha=0.55)
            sns.kdeplot(data=sub, x="_pc1", y="_pc2", ax=ax_m, color=lcolor,
                        linewidths=0.7, thresh=0.04, levels=8, alpha=0.8)
            sns.kdeplot(data=sub, x="_pc1", ax=ax_t, color=lcolor,
                        fill=True, alpha=0.45, lw=1.4, label=lname)
            sns.kdeplot(data=sub, y="_pc2", ax=ax_r, color=lcolor,
                        fill=True, alpha=0.45, lw=1.4)
    else:
        if len(pert_s) > 200:
            sns.kdeplot(data=pert_s, x="_pc1", y="_pc2", ax=ax_m,
                        cmap="mako", fill=True, thresh=0.03, levels=18, alpha=0.9)
            sns.kdeplot(data=pert_s, x="_pc1", y="_pc2", ax=ax_m,
                        color="#111111", linewidths=0.4, thresh=0.05, levels=12, alpha=0.45)
        sns.kdeplot(data=pert_s, x="_pc1", ax=ax_t, color=_THEME["pert"],
                    fill=True, alpha=0.5, lw=1.5, label="Perturbed")
        sns.kdeplot(data=ctrl_s, x="_pc1", ax=ax_t, color=_THEME["ctrl"],
                    fill=True, alpha=0.65, lw=1.5, label="Control")
        sns.kdeplot(data=pert_s, y="_pc2", ax=ax_r, color=_THEME["pert"],
                    fill=True, alpha=0.5, lw=1.5)
        sns.kdeplot(data=ctrl_s, y="_pc2", ax=ax_r, color=_THEME["ctrl"],
                    fill=True, alpha=0.65, lw=1.5)

    ax_m.scatter(ctrl_s["_pc1"], ctrl_s["_pc2"], c=_THEME["ctrl"],
                 s=8, alpha=0.8, zorder=5, rasterized=True, label="Control")
    ax_t.set_axis_off()
    ax_r.set_axis_off()
    ax_m.set_xlabel("Harmony PC1", fontsize=11)
    ax_m.set_ylabel("Harmony PC2", fontsize=11)

    if multi:
        handles = [mpatches.Patch(color=c, label=n, alpha=0.7)
                   for n, c in zip(line_names, line_colors)]
        handles.append(mpatches.Patch(color=_THEME["ctrl"], label="Control"))
    else:
        handles = [mpatches.Patch(color="#2d1e6b", alpha=0.8,
                                   label=f"Perturbed ({len(pert_s):,})"),
                   mpatches.Patch(color=_THEME["ctrl"],
                                   label=f"Control ({len(ctrl_s):,})")]
    ax_m.legend(handles=handles, loc="upper right", fontsize=9)

    n_str = f"{n_cell_lines} cell line{'s' if n_cell_lines > 1 else ''} detected"
    fig.suptitle(f"Phase 11 · Cell line detection · {n_str}\n"
                 f"{adata.n_obs:,} cells | Harmony-corrected PCA",
                 fontsize=13, fontweight="bold", y=0.97)
    path = _save(fig, cfg, "p11_cell_line_pca")
    _show(fig)
    return fig, path


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 12 · Metacell QC
# ─────────────────────────────────────────────────────────────────────────────

def plot_metacell_counts(all_meta_splits, cfg):
    _apply_theme()
    if not all_meta_splits:
        return None, None
    records = []
    for cl, splits in all_meta_splits.items():
        for sname, meta in splits.items():
            n = meta.n_obs if hasattr(meta, "n_obs") else int(meta)
            records.append({"cell_line": cl, "split": sname, "n": n})
    df = pd.DataFrame(records)
    if df.empty:
        return None, None

    splits_order = [s for s in ["train", "val", "test"] if s in df["split"].unique()]
    cell_lines   = df["cell_line"].unique().tolist()
    n_cl         = len(cell_lines)
    colors       = [_THEME["ctrl"], _THEME["pert"], _THEME["accent2"]]
    x            = np.arange(n_cl)
    width        = 0.22
    offsets      = np.linspace(-width, width, len(splits_order))

    fig, ax = plt.subplots(figsize=(max(6, n_cl * 3.5), 4.5))
    for offset, split, color in zip(offsets, splits_order, colors):
        sub  = df[df["split"] == split]
        vals = [sub[sub["cell_line"] == cl]["n"].values[0]
                if len(sub[sub["cell_line"] == cl]) > 0 else 0
                for cl in cell_lines]
        bars = ax.bar(x + offset, vals, width * 0.88, color=color,
                      edgecolor="white", lw=0.4, label=split.capitalize())
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.01,
                        f"{v:,}", ha="center", va="bottom", fontsize=8, rotation=40)
    ax.set_xticks(x)
    ax.set_xticklabels(cell_lines, fontsize=10)
    ax.set_ylabel("Metacells")
    ax.legend(loc="upper right")
    ax.set_title("A  Metacell counts per cell line × split",
                 loc="left", fontweight="bold")
    _label(ax, "A")
    path = _save(fig, cfg, "p12_metacell_counts")
    _show(fig)
    return fig, path


def plot_metacell_inner_variance(all_meta_splits, cfg):
    """
    Violin plot of within-metacell variance.
    Requires 'inner_variance' column in each metacell AnnData.obs.
    Prints a clear fix instruction if the column is absent.
    """
    _apply_theme()
    if not all_meta_splits:
        return None, None

    records = []
    missing = []
    for cl, splits in all_meta_splits.items():
        for sname, meta in splits.items():
            if not hasattr(meta, "obs"):
                continue
            if "inner_variance" not in meta.obs.columns:
                missing.append(f"{cl}/{sname}")
                continue
            vals = meta.obs["inner_variance"].dropna().values
            for v in vals:
                records.append({"cell_line": cl, "split": sname,
                                 "inner_variance": float(v),
                                 "group": f"{cl} / {sname}"})

    if missing:
        print(f"\n  ⚠ 'inner_variance' missing from: {missing}")
        print("  Fix: in Phase 12 aggregation, after building each metacell AnnData,")
        print("  compute: adata_mc.obs['inner_variance'] = per_metacell_mean_std")
        print("  (mean standard deviation of raw cell values within each metacell)\n")

    if not records:
        return None, None

    df = pd.DataFrame(records)
    groups   = sorted(df["group"].unique())
    palette  = {g: (_THEME["ctrl"] if "train" in g
                    else _THEME["pert"] if "val" in g
                    else _THEME["accent2"]) for g in groups}

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.8), 5))
    sns.violinplot(data=df, x="group", y="inner_variance", ax=ax,
                   palette=palette, inner="quartile", cut=0, lw=1.2, alpha=0.85,
                   order=groups)
    ax.set_xlabel("")
    ax.set_ylabel("Within-metacell std")
    ax.set_title("A  Metacell inner variance (low = tight, homogeneous aggregation)",
                 loc="left", fontweight="bold")
    plt.xticks(rotation=25, ha="right", fontsize=9)
    _label(ax, "A")

    fig.suptitle("Phase 12 · Metacell aggregation quality",
                 fontsize=13, fontweight="bold", y=1.02)
    path = _save(fig, cfg, "p12_inner_variance")
    _show(fig)
    return fig, path


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 13 · CHITIN diagnostics (replaces useless rectangle plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_chitin_pareto(chitin_model, cell_line, cfg):
    """Layered bivariate: Pareto sweep rank_disruption vs disc_ratio."""
    _apply_theme()
    if chitin_model is None or chitin_model.sweep_results is None:
        return None, None
    df      = chitin_model.sweep_results.copy()
    pareto  = chitin_model.pareto_front
    sel     = chitin_model.selected_params
    metrics = df["metric"].unique().tolist()
    ncols   = len(metrics)

    stab_min = df["signal_stability"].min()
    stab_rng = max(df["signal_stability"].max() - stab_min, 1e-10)

    fig = plt.figure(figsize=(6.5 * ncols, 6))
    gs  = gridspec.GridSpec(1, ncols, wspace=0.4)

    for col, metric in enumerate(metrics):
        sub = df[df["metric"] == metric].copy()
        inn = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs[col],
            width_ratios=[5, 1], height_ratios=[1, 5], wspace=0.05, hspace=0.05)
        ax_m = fig.add_subplot(inn[1, 0])
        ax_t = fig.add_subplot(inn[0, 0], sharex=ax_m)
        ax_r = fig.add_subplot(inn[1, 1], sharey=ax_m)

        norm_stab  = (sub["signal_stability"] - stab_min) / stab_rng
        npcs_vals  = sorted(sub["n_pcs"].unique())
        size_map   = {v: 40 + i * 22 for i, v in enumerate(npcs_vals)}
        sizes      = [size_map.get(int(v), 60) for v in sub["n_pcs"]]

        sc = ax_m.scatter(sub["rank_disruption"], sub["disc_ratio"],
                          c=norm_stab, cmap="rocket", s=sizes,
                          alpha=0.75, edgecolors="none", zorder=3)
        if pareto is not None:
            sp_ = pareto[pareto["metric"] == metric]
            ax_m.scatter(sp_["rank_disruption"], sp_["disc_ratio"],
                         s=[size_map.get(int(v), 60) + 20 for v in sp_["n_pcs"]],
                         facecolors="none", edgecolors=_THEME["good"],
                         lw=2, zorder=4, label="Pareto front")
        if sel and sel.get("metric") == metric:
            sr = sub[(sub["k"] == sel["k"]) & (sub["n_pcs"] == sel["n_pcs"])]
            if len(sr):
                ax_m.scatter(sr["rank_disruption"], sr["disc_ratio"],
                             marker="*", s=400, c=_THEME["accent"], zorder=6,
                             label=f"★ k={sel['k']} n_pcs={sel['n_pcs']}")

        fig.colorbar(sc, ax=ax_m, fraction=0.05, pad=0.03, label="Signal stability")
        sns.kdeplot(data=sub, x="rank_disruption", ax=ax_t,
                    color=_THEME["pert"], fill=True, alpha=0.5, lw=1.4)
        sns.kdeplot(data=sub, y="disc_ratio", ax=ax_r,
                    color=_THEME["pert"], fill=True, alpha=0.5, lw=1.4)
        ax_t.set_axis_off()
        ax_r.set_axis_off()
        ax_m.set_xlabel("Rank disruption  (1 − ρ)")
        ax_m.set_ylabel("Discrimination ratio (post/pre)")
        ax_m.legend(fontsize=8, loc="lower right")
        ax_m.set_title(f"{chr(65+col)}  [{metric}]", loc="left", fontweight="bold")

    fig.suptitle(f"CHITIN Pareto sweep · {cell_line}",
                 fontsize=13, fontweight="bold", y=1.02)
    path = _save(fig, cfg, f"p13_chitin_pareto_{cell_line}")
    _show(fig)
    return fig, path


def plot_chitin_rank_disruption(rank_corrs, cell_line, cfg):
    """Histogram + ECDF of per-gene Spearman rho."""
    _apply_theme()
    if rank_corrs is None or len(rank_corrs) == 0:
        return None, None
    mean_rho = float(np.mean(rank_corrs))
    n        = len(rank_corrs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.hist(rank_corrs, bins=60, color=_THEME["pert"],
             edgecolor="white", lw=0.3, alpha=0.85)
    ax1.axvline(mean_rho, color=_THEME["accent"], ls="--", lw=1.5,
                label=f"Mean ρ = {mean_rho:.3f}")
    ax1.axvline(1.0, color=_THEME["muted"], ls=":", lw=1.0, alpha=0.6,
                label="No disruption (ρ=1)")
    ax1.set_xlabel("Spearman ρ (pre vs post CHITIN)")
    ax1.set_ylabel("Genes")
    ax1.legend()
    ax1.set_title("A  Rank disruption histogram", loc="left", fontweight="bold")
    _label(ax1, "A")

    sorted_r = np.sort(rank_corrs)
    ecdf     = np.arange(1, n + 1) / n
    ax2.plot(sorted_r, ecdf, color=_THEME["ctrl"], lw=2.2)
    ax2.axvline(mean_rho, color=_THEME["accent"], ls="--", lw=1.5,
                label=f"Mean ρ = {mean_rho:.3f}")
    ax2.fill_betweenx([0, 1], 0, 0.92, alpha=0.07, color=_THEME["good"],
                      label="Target zone (ρ < 0.92)")
    ax2.set_xlabel("Spearman ρ")
    ax2.set_ylabel("Cumulative fraction")
    ax2.legend(fontsize=9)
    ax2.set_title("B  Rank disruption ECDF", loc="left", fontweight="bold")
    _label(ax2, "B")

    disruption_pct = float((rank_corrs < 0.92).mean() * 100)
    fig.suptitle(f"CHITIN rank disruption · {cell_line}\n"
                 f"{disruption_pct:.0f}% of genes ρ < 0.92  |  mean ρ = {mean_rho:.4f}",
                 fontsize=13, fontweight="bold", y=1.04)
    path = _save(fig, cfg, f"p13_chitin_rank_{cell_line}")
    _show(fig)
    return fig, path


def plot_chitin_discrimination(dist_pre, dist_post, cell_line, cfg):
    """Overlaid KDE: pre vs post pairwise discrimination."""
    _apply_theme()
    if dist_pre is None or dist_post is None:
        return None, None
    disc_pct = (dist_post.mean() - dist_pre.mean()) / dist_pre.mean() * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.kdeplot(dist_pre,  ax=ax, fill=True, alpha=0.45, color=_THEME["ctrl"], lw=2,
                label=f"Pre-CHITIN (μ = {dist_pre.mean():.4f})")
    sns.kdeplot(dist_post, ax=ax, fill=True, alpha=0.45, color=_THEME["pert"], lw=2,
                label=f"Post-CHITIN (μ = {dist_post.mean():.4f})")
    ax.axvline(dist_pre.mean(),  color=_THEME["ctrl"], ls="--", lw=1.3, alpha=0.8)
    ax.axvline(dist_post.mean(), color=_THEME["pert"], ls="--", lw=1.3, alpha=0.8)
    sign  = "+" if disc_pct >= 0 else ""
    color = _THEME["good"] if disc_pct >= 0 else _THEME["warn"]
    ax.text(0.97, 0.92, f"Δ discrimination: {sign}{disc_pct:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=12, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=_THEME["spine"], alpha=0.85))
    ax.set_xlabel("Cosine distance between perturbation centroids")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("A  Pairwise discrimination — pre vs post CHITIN",
                 loc="left", fontweight="bold")
    _label(ax, "A")
    fig.suptitle(f"CHITIN discrimination · {cell_line}",
                 fontsize=13, fontweight="bold", y=1.02)
    path = _save(fig, cfg, f"p13_chitin_discrimination_{cell_line}")
    _show(fig)
    return fig, path


def plot_chitin_delta_magnitudes(delta_df, delta_norms, cell_line, cfg):
    """Three-panel: delta distribution · top 25 bar · rank vs magnitude."""
    _apply_theme()
    if delta_df is None or delta_norms is None:
        return None, None

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    ax1.hist(delta_norms, bins=70, color=_THEME["pert"],
             edgecolor="white", lw=0.2, alpha=0.8, density=True)
    sns.kdeplot(delta_norms, ax=ax1, color=_THEME["accent"], lw=2.0)
    ax1.axvline(delta_norms.mean(), color=_THEME["neutral"], ls="--", lw=1.2,
                label=f"Mean = {delta_norms.mean():.2f}")
    ax1.set_xlabel("|Δ| (L2 norm)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=9)
    ax1.set_title("A  Delta magnitude distribution", loc="left", fontweight="bold")
    _label(ax1, "A")

    top25  = delta_df.head(25).copy()
    norm   = Normalize(top25["mean_delta_norm"].min(), top25["mean_delta_norm"].max())
    cmap   = plt.cm.get_cmap("rocket")
    colors = [cmap(norm(v)) for v in top25["mean_delta_norm"]]
    ax2.barh(range(len(top25)), top25["mean_delta_norm"].values,
             color=colors, edgecolor="white", lw=0.3, height=0.7)
    ax2.set_yticks(range(len(top25)))
    ax2.set_yticklabels(top25["perturbation"].values, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean |Δ|")
    ax2.set_title("B  Top 25 perturbations", loc="left", fontweight="bold")
    _label(ax2, "B")

    sorted_df = delta_df.sort_values("mean_delta_norm", ascending=False).reset_index(drop=True)
    ax3.scatter(sorted_df.index + 1, sorted_df["mean_delta_norm"],
                c=sorted_df.index, cmap="rocket_r", s=12, alpha=0.7, rasterized=True)
    ax3.set_xlabel("Perturbation rank")
    ax3.set_ylabel("Mean |Δ|")
    ax3.set_title("C  Rank vs delta magnitude", loc="left", fontweight="bold")
    _label(ax3, "C")

    fig.suptitle(f"CHITIN delta magnitudes · {cell_line}\n"
                 f"mean |Δ| = {delta_norms.mean():.3f} ± {delta_norms.std():.3f}",
                 fontsize=13, fontweight="bold", y=1.04)
    path = _save(fig, cfg, f"p13_chitin_deltas_{cell_line}")
    _show(fig)
    return fig, path


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE WATERFALL
# ─────────────────────────────────────────────────────────────────────────────

def plot_pipeline_waterfall(pipeline_tracker, cfg):
    _apply_theme()
    items = [(k, v) for k, v in pipeline_tracker.items() if isinstance(v, (int, float))]
    if len(items) < 2:
        return None, None

    labels = [k for k, _ in items]
    counts = [v for _, v in items]
    max_c  = counts[0]
    pcts   = [c / max_c for c in counts]
    cmap   = plt.cm.get_cmap("rocket")
    colors = [cmap(0.85 * p) for p in pcts]

    fig, ax = plt.subplots(figsize=(max(8, len(items) * 1.4), 5))
    bars = ax.bar(range(len(items)), counts, color=colors,
                  edgecolor="white", lw=0.4, width=0.65)
    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_c * 0.01,
                f"{count:,}\n({pct*100:.1f}%)",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Cells retained")
    ax.set_ylim(0, max_c * 1.18)
    ax.set_title("A  Cell count waterfall", loc="left", fontweight="bold")
    _label(ax, "A")

    sm = ScalarMappable(cmap="rocket", norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Fraction retained")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0%", "50%", "100%"])

    fig.suptitle("SPORE+ cell count waterfall",
                 fontsize=13, fontweight="bold", y=1.02)
    path = _save(fig, cfg, "pipeline_waterfall")
    _show(fig)
    return fig, path


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION SUMMARY (text only — a plot would waste space)
# ─────────────────────────────────────────────────────────────────────────────

def plot_detection_summary(detection_results, cfg):
    if not detection_results:
        return
    print("\n" + "─" * 55)
    print("  Phase 1 · Detection Results")
    print("─" * 55)
    for label, key in [
        ("Modality",            "modality"),
        ("Gene ID format",      "gene_id_format"),
        ("IDs harmonized",      "gene_ids_harmonized"),
        ("Combinatorial",       "is_combinatorial"),
        ("Cell line col found", "detected_cell_line_col"),
    ]:
        val  = detection_results.get(key, "—")
        flag = "✓" if val not in [None, False, "—", "not found"] else " "
        print(f"  {flag}  {label:<25}  {val}")
    print("─" * 55 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
#  Mark 2 Additions
# ─────────────────────────────────────────────────────────────────────────────

def plot_v_direction_heatmap(v_val: dict, cfg: dict):
    """
    Heatmap of pairwise cosine similarity between cell lines' V vectors.
    Only meaningful when n_cell_lines > 1.
 
    v_val structure: {"pairwise_cos": {(cl1, cl2): cos_value, ...},
                      "merge_flags": [...]}
    """
    _apply_theme()
    pairwise = v_val.get("pairwise_cos", {})
    if not pairwise:
        return None, None
 
    import pandas as pd
    lines = sorted(set(k for pair in pairwise.keys() for k in pair))
    n     = len(lines)
    mat   = np.eye(n)
    for (a, b), val in pairwise.items():
        i = lines.index(a)
        j = lines.index(b)
        mat[i, j] = val
        mat[j, i] = val
 
    df_mat = pd.DataFrame(mat, index=lines, columns=lines)
 
    fig, ax = plt.subplots(figsize=(max(5, n * 0.9), max(4.5, n * 0.85)))
    sns.heatmap(df_mat, ax=ax, cmap="rocket_r", vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=0.5,
                linecolor=_THEME["spine"],
                cbar_kws={"label": "Cosine similarity of V vectors",
                           "shrink": 0.8})
    ax.set_title("A  Cell line systematic variation alignment\n"
                 "(values near 1 = shared systematic direction)",
                 loc="left", fontweight="bold")
    _label(ax, "A")
 
    merge_flags = v_val.get("merge_flags", [])
    if merge_flags:
        ax.text(0.98, 0.02,
                f"Merge flags: {', '.join(str(f) for f in merge_flags)}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color=_THEME["warn"])
 
    fig.suptitle("Phase 11 · V-direction alignment across cell lines",
                 fontsize=13, fontweight="bold", y=1.02)
    path = _save(fig, cfg, "p11_v_direction_heatmap")
    _show(fig)
    return fig, path
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Paste into plotting.py at the end of PHASE 13 / CHITIN section
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_chitin_summary(chitin_summary: dict, cfg: dict):
    """
    Wrapper: calls the three CHITIN diagnostic plots for every cell line
    in chitin_summary. Replaces the old useless rectangle plot.
 
    chitin_summary structure:
        {cell_line: {"rank_corrs": np.ndarray,
                     "dist_pre": np.ndarray,
                     "dist_post": np.ndarray,
                     "delta_df": pd.DataFrame,
                     "delta_norms": np.ndarray,
                     "chitin_model": ChitinModel}}
    """
    if not chitin_summary:
        print("  plot_chitin_summary: no CHITIN summary data to plot")
        return None, None
 
    paths = []
    for cell_line, cs in chitin_summary.items():
        rank_corrs  = cs.get("rank_corrs")
        dist_pre    = cs.get("dist_pre")
        dist_post   = cs.get("dist_post")
        delta_df    = cs.get("delta_df")
        delta_norms = cs.get("delta_norms")
        model       = cs.get("chitin_model")
 
        if rank_corrs is not None and len(rank_corrs) > 0:
            _, p = plot_chitin_rank_disruption(rank_corrs, cell_line, cfg)
            if p:
                paths.append(p)
 
        if dist_pre is not None and dist_post is not None:
            _, p = plot_chitin_discrimination(dist_pre, dist_post, cell_line, cfg)
            if p:
                paths.append(p)
 
        if delta_df is not None and delta_norms is not None:
            _, p = plot_chitin_delta_magnitudes(delta_df, delta_norms, cell_line, cfg)
            if p:
                paths.append(p)
 
        if model is not None and model.sweep_results is not None:
            _, p = plot_chitin_pareto(model, cell_line, cfg)
            if p:
                paths.append(p)
 
    return None, paths
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  M2 · Doublet detection QC plot
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_doublet_summary(adata, cfg: dict):
    """
    Two-panel:
    A  Histogram of doublet scores (all cells), threshold marked
    B  ECDF of doublet scores — highlights where predicted doublets fall
    """
    _apply_theme()
    if "doublet_score" not in adata.obs.columns:
        print("  No doublet_score in obs — run Phase 4 first")
        return None, None
 
    scores     = adata.obs["doublet_score"].values.astype(float)
    predicted  = adata.obs.get("predicted_doublet",
                                np.zeros(len(scores), dtype=bool)).values.astype(bool)
    n_doublets = int(predicted.sum())
 
    # Infer threshold from the minimum score among predicted doublets
    threshold = float(scores[predicted].min()) if n_doublets > 0 else 0.5
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
 
    # A: score histogram
    ax1.hist(scores[~predicted], bins=60, color=_THEME["ctrl"],
             edgecolor="white", lw=0.2, alpha=0.8,
             label=f"Singlets ({(~predicted).sum():,})")
    ax1.hist(scores[predicted], bins=60, color=_THEME["accent"],
             edgecolor="white", lw=0.2, alpha=0.8,
             label=f"Doublets ({n_doublets:,})")
    ax1.axvline(threshold, color=_THEME["neutral"], ls="--", lw=1.5,
                label=f"Threshold: {threshold:.3f}")
    ax1.set_xlabel("Doublet score")
    ax1.set_ylabel("Cells")
    ax1.legend(fontsize=9)
    ax1.set_title("A  Doublet score distribution", loc="left", fontweight="bold")
    _label(ax1, "A")
 
    # B: ECDF
    sorted_s = np.sort(scores)
    ecdf     = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    ax2.plot(sorted_s, ecdf, color=_THEME["ctrl"], lw=2)
    ax2.axvline(threshold, color=_THEME["accent"], ls="--", lw=1.5,
                label=f"Threshold ({n_doublets:,} doublets = "
                      f"{n_doublets/max(len(scores),1)*100:.1f}%)")
    ax2.set_xlabel("Doublet score")
    ax2.set_ylabel("Cumulative fraction")
    ax2.legend(fontsize=9)
    ax2.set_title("B  Doublet score ECDF", loc="left", fontweight="bold")
    _label(ax2, "B")
 
    fig.suptitle(f"Phase 4 · Doublet detection\n"
                 f"{n_doublets:,} / {len(scores):,} predicted doublets "
                 f"({n_doublets/max(len(scores),1)*100:.2f}%)",
                 fontsize=13, fontweight="bold", y=1.04)
    path = _save(fig, cfg, "p04_doublet_summary")
    _show(fig)
    return fig, path
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  M2 · Ambient RNA QC plot
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_ambient_summary(adata, cfg: dict):
    """
    Two-panel:
    A  Scatter: ambient score vs total_counts (reveals the ambient cloud)
    B  Histogram of ambient scores with flag threshold marked
    """
    _apply_theme()
    if "ambient_score" not in adata.obs.columns:
        print("  No ambient_score in obs — run Phase 3 first")
        return None, None
 
    scores   = adata.obs["ambient_score"].values.astype(float)
    flagged  = adata.obs.get("ambient_flagged",
                              np.zeros(len(scores), dtype=bool)).values.astype(bool)
    n_flag   = int(flagged.sum())
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
 
    # A: ambient vs UMI scatter
    if "total_counts" in adata.obs.columns:
        umi = adata.obs["total_counts"].values.astype(float)
        # Subsample for speed
        rng    = np.random.default_rng(42)
        n_show = min(50_000, len(scores))
        idx    = rng.choice(len(scores), n_show, replace=False)
        ax1.scatter(umi[idx], scores[idx], c=flagged[idx].astype(int),
                    cmap="RdYlGn_r", s=3, alpha=0.5, rasterized=True)
        ax1.set_xscale("log")
        ax1.set_xlabel("Total UMI counts (log)")
        ax1.set_ylabel("Ambient score")
        ax1.set_title("A  Ambient score vs UMI — flagged cells in red",
                      loc="left", fontweight="bold")
        _label(ax1, "A")
    else:
        ax1.set_visible(False)
 
    # B: score histogram
    threshold = float(scores[flagged].min()) if n_flag > 0 else 0.8
    ax2.hist(scores[~flagged], bins=60, color=_THEME["ctrl"],
             edgecolor="white", lw=0.2, alpha=0.8,
             label=f"Clean ({(~flagged).sum():,})")
    if n_flag > 0:
        ax2.hist(scores[flagged], bins=40, color=_THEME["accent"],
                 edgecolor="white", lw=0.2, alpha=0.8,
                 label=f"Flagged ({n_flag:,})")
    ax2.axvline(threshold, color=_THEME["neutral"], ls="--", lw=1.5,
                label=f"Threshold: {threshold:.2f}")
    ax2.set_xlabel("Ambient score")
    ax2.set_ylabel("Cells")
    ax2.legend(fontsize=9)
    ax2.set_title("B  Ambient score distribution", loc="left", fontweight="bold")
    _label(ax2, "B")
 
    fig.suptitle(f"Phase 3 · Ambient RNA scoring\n"
                 f"{n_flag:,} / {len(scores):,} cells flagged as potentially ambient",
                 fontsize=13, fontweight="bold", y=1.04)
    path = _save(fig, cfg, "p03_ambient_summary")
    _show(fig)
    return fig, path
