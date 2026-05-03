"""
SPORE+ · src/phase11_cell_line_separation.py
─────────────────────────────────────────────
Phase 11: Cell Line Detection & Separation

This phase identifies distinct cell line populations and adds a
'sporeplus_cell_line' column to .obs. It uses a 3-tier approach:

  Tier 1 (labeled):   .obs already has a cell line column (detected in Phase 1)
  Tier 2 (hinted):    expected_n_cell_lines known → k-means on Harmony PCA
  Tier 3 (automatic): HDBSCAN + GMM BIC sweep + bootstrap stability validation

After labeling, a V-direction validation checks whether detected populations
have genuinely divergent regulatory contexts (different regulatory programs)
or just differ in state/intensity (same program, don't split).

MEMORY SAFETY:
  All clustering operates on adata.obsm['X_pca_harmony'] (or 'X_pca'),
  which is a compact float32 matrix. Never loads adata.X for clustering.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from .utils import log_phase_header, snapshot, log_memory, force_gc


# ═══════════════════════════════════════════════════════════════════════════
#  TIER 1: USE EXISTING LABELS
# ═══════════════════════════════════════════════════════════════════════════

def _tier1_use_existing_labels(adata, cell_line_col: str, logger) -> pd.Series:
    """Use existing .obs cell line labels directly."""
    labels = adata.obs[cell_line_col].astype(str)
    unique = sorted(labels.unique().tolist())
    logger.info(
        f"  Tier 1: using existing '{cell_line_col}' column → "
        f"{len(unique)} cell lines: {unique[:8]}"
        f"{'...' if len(unique) > 8 else ''}")
    return labels


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER: GET PCA EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════

def _get_embedding(adata, logger):
    """
    Retrieve the best available PCA embedding for clustering.
    Preference: X_pca_harmony > X_pca.
    """
    if "X_pca_harmony" in adata.obsm:
        logger.info("  Using Harmony-corrected PCA for cell line clustering")
        return adata.obsm["X_pca_harmony"].astype(np.float32)
    elif "X_pca_chitin" in adata.obsm:
        logger.info("  Using CHITIN PCA for cell line clustering")
        return adata.obsm["X_pca_chitin"].astype(np.float32)
    elif "X_pca" in adata.obsm:
        logger.info("  Using standard PCA for cell line clustering")
        return adata.obsm["X_pca"].astype(np.float32)
    else:
        logger.warning(
            "  No PCA embedding found. "
            "Run Phase 10 (confounder mitigation) before Phase 11.")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  TIER 2: HINT-GUIDED K-MEANS
# ═══════════════════════════════════════════════════════════════════════════

def _tier2_hint_guided_kmeans(adata, n_cell_lines: int, logger) -> pd.Series:
    """
    Run k-means with a user-specified K on the Harmony PCA embedding.
    Fast and reliable when the number of cell lines is known.
    """
    from sklearn.cluster import MiniBatchKMeans

    emb = _get_embedding(adata, logger)
    if emb is None:
        return None

    logger.info(
        f"  Tier 2: hint-guided k-means with k={n_cell_lines} "
        f"on {adata.n_obs:,} cells...")

    n_pcs = min(30, emb.shape[1])
    emb_sub = emb[:, :n_pcs]

    km = MiniBatchKMeans(
        n_clusters=n_cell_lines, random_state=42, n_init="auto",
        max_iter=300, batch_size=min(10000, adata.n_obs))
    labels_int = km.fit_predict(emb_sub)
    labels     = pd.Series(
        [f"cell_line_{i+1}" for i in labels_int],
        index=adata.obs_names, dtype="category")

    for lbl in sorted(labels.unique()):
        n = (labels == lbl).sum()
        logger.info(f"    {lbl}: {n:,} cells")
    return labels


# ═══════════════════════════════════════════════════════════════════════════
#  TIER 3: AUTOMATIC DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _compute_gmm_bic_sweep(emb_sub, max_k: int, logger):
    """
    Fit GMMs for K=1..max_k, return BIC scores per K.
    Lower BIC = better fit. The elbow/minimum is the best K.
    """
    from sklearn.mixture import GaussianMixture
    try:
        from tqdm.auto import tqdm as _tqdm
        _use_tqdm = True
    except ImportError:
        _use_tqdm = False

    bic_scores = []
    k_range = range(1, max_k + 1)
    if _use_tqdm:
        k_range = _tqdm(list(k_range), desc="GMM BIC sweep", leave=False)

    for k in k_range:
        try:
            gm = GaussianMixture(
                n_components=k, covariance_type="full",
                random_state=42, max_iter=200, n_init=2)
            gm.fit(emb_sub)
            bic = gm.bic(emb_sub)
            bic_scores.append((k, bic, gm))
            logger.info(f"    K={k}: BIC={bic:.1f}")
        except Exception as e:
            logger.warning(f"    K={k}: GMM failed ({e})")
            bic_scores.append((k, np.inf, None))

    return bic_scores


def _find_bic_elbow(bic_scores, min_bic_improvement: float = 50.0):
    """
    Find the optimal K from BIC scores.
    Select the smallest K where adding K+1 improves BIC by less than
    min_bic_improvement (diminishing returns elbow).
    Falls back to the global minimum if no clear elbow is found.
    """
    ks   = [s[0] for s in bic_scores]
    bics = [s[1] for s in bic_scores]

    best_k = ks[np.argmin(bics)]

    # Check for elbow: first K where improvement plateaus
    for i in range(len(bics) - 1):
        improvement = bics[i] - bics[i + 1]
        if improvement < min_bic_improvement:
            return ks[i], bics[i], True  # (k, bic, is_elbow)

    return best_k, bics[np.argmin(bics)], False


def _bootstrap_stability(emb_sub, k: int, n_bootstrap: int,
                          bootstrap_frac: float, logger):
    """
    Assess cluster stability via bootstrap resampling.
    Fits k-means n_bootstrap times on 80% subsamples.
    Returns mean Adjusted Rand Index across bootstrap pairs.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score

    n_cells = emb_sub.shape[0]
    n_sub   = int(n_cells * bootstrap_frac)
    rng     = np.random.default_rng(42)
    labels_all = []

    try:
        from tqdm.auto import tqdm as _tqdm
        boot_iter = _tqdm(range(n_bootstrap), desc=f"  Bootstrap k={k}", leave=False)
    except ImportError:
        boot_iter = range(n_bootstrap)

    for i in boot_iter:
        idx = rng.choice(n_cells, size=n_sub, replace=False)
        km  = MiniBatchKMeans(
            n_clusters=k, random_state=i, n_init="auto", max_iter=200)
        lbls = km.fit_predict(emb_sub[idx])
        labels_all.append((idx, lbls))

    # Compare all pairs of bootstrap runs
    aris = []
    for i in range(min(10, len(labels_all))):  # compare first 10 pairs for speed
        for j in range(i + 1, min(10, len(labels_all))):
            idx_i, lbl_i = labels_all[i]
            idx_j, lbl_j = labels_all[j]
            common = np.intersect1d(idx_i, idx_j)
            if len(common) < 10:
                continue
            # Get labels for common cells
            mask_i = np.isin(idx_i, common)
            mask_j = np.isin(idx_j, common)
            aris.append(adjusted_rand_score(
                lbl_i[mask_i], lbl_j[mask_j]))

    mean_ari = float(np.mean(aris)) if aris else 0.0
    logger.info(
        f"  Bootstrap stability (K={k}, {n_bootstrap} runs): "
        f"mean ARI = {mean_ari:.3f}")
    return mean_ari


def _tier3_automatic_detection(adata, p11: dict, logger):
    """
    Fully automatic cell line detection:
    1. HDBSCAN to get an initial baseline K estimate
    2. GMM BIC sweep up to 2× HDBSCAN K
    3. Bootstrap stability at the best K
    4. Report confidence and flag ambiguous cases

    Returns: (labels: pd.Series, n_detected: int, confidence: float, metadata: dict)
    """
    auto_cfg = p11.get("auto_detect", {})
    max_k    = p11.get("auto_detect", {}).get("max_k_to_try", 15)
    n_boot   = auto_cfg.get("n_bootstrap", 50)
    boot_f   = auto_cfg.get("bootstrap_frac", 0.80)
    min_ari  = auto_cfg.get("min_stability_ari", 0.85)
    min_bic  = auto_cfg.get("min_bic_improvement", 50.0)

    emb = _get_embedding(adata, logger)
    if emb is None:
        return None, 1, 0.0, {}

    n_pcs    = min(30, emb.shape[1])
    emb_sub  = emb[:, :n_pcs]
    metadata = {}

    # Step 1: HDBSCAN for initial K estimate
    # CRITICAL: always subsample before HDBSCAN.
    # HDBSCAN is O(n^1.5) and single-threaded. On 1.5M cells it takes 4-20 hours.
    # On 50k cells it takes ~30 seconds and gives the same K estimate.
    # After HDBSCAN determines K, MiniBatchKMeans (parallelizable, fast)
    # assigns the full dataset.
    MAX_HDBSCAN_CELLS = auto_cfg.get("max_hdbscan_cells", 50_000)
    hdbscan_k = None
    try:
        import hdbscan
        n_hdbscan = min(MAX_HDBSCAN_CELLS, adata.n_obs)
        if n_hdbscan < adata.n_obs:
            rng_hdb   = np.random.default_rng(42)
            hdb_idx   = rng_hdb.choice(adata.n_obs, size=n_hdbscan, replace=False)
            emb_hdbscan = emb_sub[hdb_idx]
            logger.info(
                f"  HDBSCAN: subsampled to {n_hdbscan:,}/{adata.n_obs:,} cells "
                f"(full matrix would take hours)")
        else:
            emb_hdbscan = emb_sub
            logger.info(f"  HDBSCAN: using all {adata.n_obs:,} cells")

        min_cluster_size = max(50, int(n_hdbscan * 0.01))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=10,
            metric="euclidean", core_dist_n_jobs=1)
        clusterer.fit(emb_hdbscan)
        n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        hdbscan_k  = max(1, n_clusters)
        noise_frac = (clusterer.labels_ == -1).mean()
        logger.info(
            f"  HDBSCAN: {hdbscan_k} clusters, "
            f"{noise_frac*100:.1f}% noise points "
            f"(min_cluster_size={min_cluster_size}, n_cells={n_hdbscan:,})")
        metadata["hdbscan_k"] = hdbscan_k
        metadata["hdbscan_noise_frac"] = float(noise_frac)
        metadata["hdbscan_n_cells_used"] = n_hdbscan
    except ImportError:
        logger.warning(
            "  hdbscan not installed. Skipping HDBSCAN step. "
            "Run: pip install hdbscan --break-system-packages")
        hdbscan_k = None

    # Step 2: GMM BIC sweep
    # Cap the sweep at max_k, or 2× HDBSCAN K if available
    sweep_max_k = min(max_k, (hdbscan_k * 2 if hdbscan_k else max_k))
    sweep_max_k = max(sweep_max_k, 2)  # always test at least K=1,2

    # Subsample for GMM speed (GMM is O(n²) in n_cells)
    n_gmm_cells = min(5000, adata.n_obs)
    rng         = np.random.default_rng(42)
    gmm_idx     = rng.choice(adata.n_obs, size=n_gmm_cells, replace=False)
    emb_gmm     = emb_sub[gmm_idx]

    logger.info(f"  GMM BIC sweep: K=1..{sweep_max_k} on {n_gmm_cells:,} cells")
    bic_scores   = _compute_gmm_bic_sweep(emb_gmm, sweep_max_k, logger)
    best_k, best_bic, is_elbow = _find_bic_elbow(bic_scores, min_bic)
    metadata["gmm_bic_best_k"]  = best_k
    metadata["gmm_bic_is_elbow"] = is_elbow

    logger.info(
        f"  GMM BIC elbow: K={best_k} "
        f"({'clear elbow' if is_elbow else 'global minimum — no clear elbow'})")

    if not is_elbow:
        logger.warning(
            f"  ⚠ No clear BIC elbow found. K={best_k} is the global BIC minimum "
            f"but cell line boundaries may be ambiguous. "
            f"Consider providing expected_n_cell_lines in config.")

    # Step 3: Bootstrap stability at best K
    if best_k > 1:
        mean_ari = _bootstrap_stability(emb_sub, best_k, n_boot, boot_f, logger)
        metadata["bootstrap_ari"] = mean_ari

        if mean_ari < min_ari:
            logger.warning(
                f"  ⚠ Bootstrap stability ARI={mean_ari:.3f} < threshold {min_ari}. "
                f"Clusters may not be robust. Manual review recommended.")
            confidence = mean_ari
        else:
            confidence = mean_ari
            logger.info(f"  ✓ Clusters are stable (ARI={mean_ari:.3f})")
    else:
        confidence = 1.0
        metadata["bootstrap_ari"] = 1.0

    # Step 4: Fit final k-means with best K on full dataset
    if best_k == 1:
        labels = pd.Series(
            ["cell_line_1"] * adata.n_obs,
            index=adata.obs_names, dtype="category")
        logger.info("  Tier 3: K=1 detected — single cell line dataset")
    else:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(
            n_clusters=best_k, random_state=42, n_init="auto", max_iter=300,
            batch_size=min(10000, adata.n_obs))
        labels_int = km.fit_predict(emb_sub)
        labels = pd.Series(
            [f"cell_line_{i+1}" for i in labels_int],
            index=adata.obs_names, dtype="category")

        for lbl in sorted(labels.unique()):
            n = (labels == lbl).sum()
            logger.info(f"    {lbl}: {n:,} cells ({n/adata.n_obs*100:.1f}%)")

    return labels, best_k, confidence, metadata


# ═══════════════════════════════════════════════════════════════════════════
#  V-DIRECTION VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def _v_direction_validation(adata, cell_line_labels, cfg, logger):
    """
    For each detected cell line, compute V = O_pert - C_ctrl in gene space.
    Compare V vectors across cell lines.

    If cosine similarity between any two cell lines' V vectors > threshold:
    → Same regulatory program, likely metabolic state or batch artifact.
    → Flag for potential merge.

    If cosine similarity < threshold:
    → Divergent regulatory programs → genuine cell lines → split warranted.

    Returns: validation_results dict with per-cell-line V vectors and
    pairwise cosine similarities.
    """
    p11         = cfg.get("phase11_cell_line", {})
    cos_thresh  = p11.get("v_direction_angle_threshold", 0.5)
    pert_col    = cfg.get("dataset", {}).get("perturbation_col", "gene")
    ctrl_label  = cfg.get("dataset", {}).get("control_label", "non-targeting")

    import scipy.sparse as sp
    unique_lines = sorted(cell_line_labels.unique().tolist())

    V_vectors = {}
    for line in unique_lines:
        line_mask  = cell_line_labels == line
        line_adata = adata[line_mask]

        pert_values = line_adata.obs[pert_col].values
        ctrl_mask_l = pert_values == ctrl_label
        pert_mask_l = ~ctrl_mask_l

        if ctrl_mask_l.sum() < 10 or pert_mask_l.sum() < 10:
            logger.warning(
                f"  V-direction: '{line}' has too few ctrl ({ctrl_mask_l.sum()}) "
                f"or pert ({pert_mask_l.sum()}) cells — skipping")
            continue

        X = line_adata.X
        if sp.issparse(X):
            C_ctrl = np.asarray(X[ctrl_mask_l].mean(axis=0)).flatten()
            O_pert = np.asarray(X[pert_mask_l].mean(axis=0)).flatten()
        else:
            C_ctrl = X[ctrl_mask_l].mean(axis=0)
            O_pert = X[pert_mask_l].mean(axis=0)

        V = O_pert - C_ctrl
        V_norm = np.linalg.norm(V)
        if V_norm > 1e-10:
            V_vectors[line] = V / V_norm
        else:
            logger.warning(f"  V-direction: '{line}' V is near-zero")

    # Compute pairwise cosines between all V vectors
    line_names = list(V_vectors.keys())
    n = len(line_names)
    cosines = {}
    merge_flags = []

    for i in range(n):
        for j in range(i + 1, n):
            name_i, name_j = line_names[i], line_names[j]
            cos = float(np.dot(V_vectors[name_i], V_vectors[name_j]))
            cosines[(name_i, name_j)] = cos
            if cos > cos_thresh:
                merge_flags.append((name_i, name_j, cos))
                logger.warning(
                    f"  V-direction: '{name_i}' ↔ '{name_j}' "
                    f"cos={cos:.3f} > {cos_thresh} → PARALLEL programs. "
                    f"May be same cell line in different states.")
            else:
                logger.info(
                    f"  V-direction: '{name_i}' ↔ '{name_j}' "
                    f"cos={cos:.3f} → DIVERGENT programs. Split warranted.")

    validation = {
        "V_vectors":    V_vectors,
        "pairwise_cos": cosines,
        "merge_flags":  merge_flags,
        "n_merge_flagged": len(merge_flags),
    }
    return validation


# ═══════════════════════════════════════════════════════════════════════════
#  RUN PHASE 11
# ═══════════════════════════════════════════════════════════════════════════

def run_phase11(adata, cfg: dict, logger):
    """
    Run cell line detection and add 'sporeplus_cell_line' to .obs.

    Returns: (adata_with_cell_line_label, detection_meta dict)
    """
    log_phase_header(logger, 11, "Cell Line Detection & Separation")
    p11 = cfg.get("phase11_cell_line", {})

    enabled = p11.get("enabled", "auto")
    if enabled is False:
        logger.info("  Phase 11: DISABLED in config")
        adata.obs["sporeplus_cell_line"] = "single_line"
        return adata, {"skipped": True, "n_cell_lines": 1}

    log_memory(logger, "Phase 11 start")
    detection_meta = {}

    # ── Determine which tier to use ────────────────────────────────────────
    cell_line_col  = p11.get("cell_line_col")   # populated by Phase 1 Tier 1
    expected_n     = p11.get("expected_n_cell_lines")

    if cell_line_col and cell_line_col in adata.obs.columns:
        # ── TIER 1: existing labels ────────────────────────────────────────
        logger.info(f"  → Tier 1: using existing column '{cell_line_col}'")
        labels = _tier1_use_existing_labels(adata, cell_line_col, logger)
        detection_meta["tier_used"] = 1
        detection_meta["source_col"] = cell_line_col

    elif expected_n and int(expected_n) > 0:
        # ── TIER 2: hint-guided k-means ────────────────────────────────────
        logger.info(f"  → Tier 2: hint-guided k-means (expected K={expected_n})")
        labels = _tier2_hint_guided_kmeans(adata, int(expected_n), logger)
        if labels is None:
            logger.warning("  Tier 2 failed (no PCA embedding). Defaulting to single line.")
            adata.obs["sporeplus_cell_line"] = "single_line"
            return adata, {"error": "no_embedding", "n_cell_lines": 1}
        detection_meta["tier_used"] = 2
        detection_meta["expected_n"] = expected_n

    elif p11.get("auto_detect", {}).get("enabled", True):
        # ── TIER 3: fully automatic ────────────────────────────────────────
        logger.info("  → Tier 3: fully automatic detection")
        labels, n_detected, confidence, auto_meta = _tier3_automatic_detection(
            adata, p11, logger)
        if labels is None:
            logger.warning("  Tier 3 failed. Defaulting to single line.")
            adata.obs["sporeplus_cell_line"] = "single_line"
            return adata, {"error": "tier3_failed", "n_cell_lines": 1}
        detection_meta.update({
            "tier_used": 3,
            "n_detected": n_detected,
            "confidence": confidence,
            **auto_meta,
        })
    else:
        logger.info("  No detection method available. Treating as single cell line.")
        adata.obs["sporeplus_cell_line"] = "single_line"
        return adata, {"n_cell_lines": 1}

    # ── Apply labels to .obs ───────────────────────────────────────────────
    adata.obs["sporeplus_cell_line"] = labels.values
    unique_lines = sorted(labels.unique().tolist())
    n_lines = len(unique_lines)
    detection_meta["cell_lines"] = unique_lines
    detection_meta["n_cell_lines"] = n_lines

    logger.info(f"  {n_lines} cell line(s) detected: {unique_lines[:8]}")

    # ── Filter cell lines with too few cells ───────────────────────────────
    min_cells = p11.get("min_cells_per_cell_line", 500)
    line_counts = adata.obs["sporeplus_cell_line"].value_counts()
    small_lines = line_counts[line_counts < min_cells].index.tolist()
    if small_lines:
        logger.warning(
            f"  ⚠ {len(small_lines)} cell line(s) have < {min_cells} cells "
            f"and will be EXCLUDED from Phase 12 output: {small_lines}")
        detection_meta["excluded_small_lines"] = small_lines

    # ── V-direction validation ─────────────────────────────────────────────
    if p11.get("v_direction_validation", True) and n_lines > 1:
        logger.info("  Running V-direction validation...")
        try:
            v_valid = _v_direction_validation(adata, labels, cfg, logger)
            detection_meta["v_validation"] = {
                "pairwise_cos": {
                    f"{k[0]}_{k[1]}": v
                    for k, v in v_valid["pairwise_cos"].items()},
                "n_merge_flagged": v_valid["n_merge_flagged"],
            }
            if v_valid["n_merge_flagged"] > 0:
                logger.warning(
                    f"  ⚠ {v_valid['n_merge_flagged']} cell line pair(s) have parallel "
                    f"V vectors. These may be the same cell line in different states "
                    f"rather than genuinely distinct cell lines.")
        except Exception as e:
            logger.warning(f"  V-direction validation failed: {e}")

    # ── auto mode: if single cell line detected, effectively a no-op ───────
    if enabled == "auto" and n_lines == 1:
        logger.info(
            "  Single cell line detected — Phase 12 will produce "
            "standard (non-split) output.")

    snapshot(adata, "Post Phase 11 Cell Line Detection", logger)
    log_memory(logger, "Phase 11 end")
    return adata, detection_meta
