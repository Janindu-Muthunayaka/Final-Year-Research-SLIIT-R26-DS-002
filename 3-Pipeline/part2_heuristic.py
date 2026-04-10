# =============================================================================
# part2_heuristic.py  —  Phase 2: Statistical Pattern Search & Multi-Output RF
#
# Purpose
# -------
# Uses the per-image CER table from Part 1 to find, per image, a parameter
# combo that beats any grid point — by probing the 4-D parameter space in a
# data-driven, correlation-weighted coordinate-descent pattern.
#
# How the search works (three-phase, ≤ MAX_HOPS pipeline runs per image)
# -----------------------------------------------------------------------
# PHASE A — Global Anchor Ranking  (free — no pipeline runs)
#   1. Run a "Global Census" on per_image_cer.csv: for every grid point, count
#      how many of the 250 images achieved their personal best CER there.
#   2. For each image, find all grid combos tied at the minimum CER.
#   3. Tie-break by (a) highest global frequency, then (b) highest
#      multi_seg_threshold (Pearson r = -0.82 → higher is better).
#   The result is a single, robustly chosen "center" point.
#
# PHASE B — Weighted Coordinate Probing  (≤ MAX_HOPS pipeline runs)
#   Parameters are probed in influence order, derived from correlation_matrix.csv
#   and param_relationship_matrix.csv read at runtime from Part 1 outputs:
#     1. multi_seg_threshold first  (|r| = 0.82 — dominant influence)
#     2. window_pad  (coupled to smoothing_k via 0.91 relationship score;
#        they are moved together as a diagonal pair)
#     3. close_k  (low influence — probed last, large step)
#   For each parameter (or coupled pair), we probe +step and -step from the
#   current center and keep the best result.
#
# PHASE C — Decaying Step Convergence
#   After each full coordinate sweep:
#     • If any probe improved the best CER → update center, repeat.
#     • If NO probe improved → halve ALL step sizes and repeat.
#     • Exit when the step size for multi_seg_threshold drops below 0.1.
#   This gives "precision focus" around the optimum without wasting budget
#   on low-yield probes.
#
# Why this beats the old binary-search along A→B
# -----------------------------------------------
# • Old code: 1-D search on a line between two grid points → misses diagonal
#   optima; odd-integer snapping causes repeated identical probes (0% gain).
# • New code: 4-D coordinate descent, influence-weighted, with diagonal
#   coupling between correlated parameters; off-grid decimals explored fully.
# • The RF training receives genuinely diverse, off-grid parameter examples.
#
# What it does
# ------------
# 1. Loads per_image_cer.csv, correlation_matrix.csv, and
#    param_relationship_matrix.csv from Part 1.
# 2. Computes a global grid popularity census.
# 3. Selects the same MAPPING_SAMPLE images (same seed as Part 1).
# 4. Extracts 14 image statistics from each raw image.
# 5. For each image, runs the three-phase pattern search above.
# 6. Writes live CSVs after every hop / image.
# 7. Trains a MultiOutputClassifier (RandomForest) on the results.
# 8. Saves all artefacts for Part 3.
#
# Usage
# -----
#   python part2_heuristic.py
#   python part2_heuristic.py --sensitivity_dir path/to/phase1_sensitivity
#   python part2_heuristic.py --sample 30 --seed 42 --max_hops 10
# =============================================================================

from __future__ import annotations

import os
import sys
import json
import csv
import time
import argparse
import warnings
import random as _random
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import cv2

warnings.filterwarnings("ignore")

# ── Make pipeline_core importable ────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pipeline_core import (
    PipelineConfig,
    run_pipeline,
    _load_model,
    _load_labels,
    _DEVICE,
    INPUT_FOLDER, LABEL_CSV, MODEL_PATH, CLASS_MAP, WORK_ROOT,
    S1_WORKERS,
)

import torch

# =============================================================================
# TOP-OF-FILE CONFIGURATION  (adjust here)
# =============================================================================

PHASE2_OUT      = os.path.join(WORK_ROOT, "Info", "phase2_heuristic")
SENSITIVITY_DIR = os.path.join(WORK_ROOT, "Info", "phase1_sensitivity")

# ── Must match Part 1 so the same images are used ────────────────────────────
MAPPING_SAMPLE = 250   # ← keep in sync with SENS_SAMPLE in part1
MAPPING_SEED   = 42    # ← keep in sync with SENS_SEED   in part1

# ── Search budget ─────────────────────────────────────────────────────────────
MAX_HOPS      = 10    # max real pipeline runs per image in phase B+C
MIN_CER_DELTA = 0.1   # minimum gain to count as "improvement" (avoids noise)

# ── Model training ────────────────────────────────────────────────────────────
TEST_SPLIT = 0.2

# =============================================================================
# PARAMETER GRID  (must match Part 1)
# =============================================================================

SWEPT_GRID = {
    "smoothing_k":         [1, 3, 5],
    "close_k":             [2, 3, 5],
    "window_pad":          [4, 8, 12, 20],
    "multi_seg_threshold": [95.0, 97.0, 99.0],
}

# Physical limits used when clamping continuous probes
PARAM_LIMITS = {
    "smoothing_k":         (1, 9),
    "close_k":             (1, 9),
    "window_pad":          (2, 40),
    "multi_seg_threshold": (80.0, 100.0),
}

SWEPT_PARAMS = ["smoothing_k", "close_k", "window_pad", "multi_seg_threshold"]

FIXED_PARAMS = {
    "target_height":    384,
    "skeleton_dil":     1,
    "valley_min_width": 2,
    "word_gap_px":      35,
}

# =============================================================================
# DEFAULT SEARCH CONFIGURATION
# (overridden at runtime by values read from Part 1 outputs)
# =============================================================================

# Influence order: index 0 = most influential parameter
# Overridden in _build_search_config() using correlation_matrix.csv
DEFAULT_INFLUENCE_ORDER = [
    "multi_seg_threshold",  # |r| = 0.82
    "window_pad",           # |r| = 0.29
    "close_k",              # |r| = 0.17
    "smoothing_k",          # |r| = 0.03
]

# Initial step sizes per parameter (tuned to grid spacing)
DEFAULT_INITIAL_STEPS = {
    "smoothing_k":         2.0,   # odd-int grid: 1,3,5 → step of 2
    "close_k":             1.0,
    "window_pad":          4.0,   # grid: 4,8,12,20 → step of 4
    "multi_seg_threshold": 1.0,   # grid: 95,97,99 → step of 1 (fine)
}

# Convergence threshold for multi_seg_threshold step size
CONVERGENCE_STEP_THRESHOLD = 0.1

# =============================================================================
# HELPERS
# =============================================================================

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _open_csv_appender(path: str, fieldnames: list[str]) -> tuple:
    is_new = not os.path.exists(path)
    fh     = open(path, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    if is_new:
        writer.writeheader()
        fh.flush()
    return fh, writer


def combo_key(combo: dict) -> tuple:
    return tuple(float(combo[k]) for k in SWEPT_PARAMS)


def clamp_and_round(combo: dict) -> dict:
    """
    Clamp each parameter to its physical limits and round to a legal value:
      smoothing_k, close_k → nearest odd integer ≥ 1
      window_pad            → nearest integer
      multi_seg_threshold   → float, 1 decimal place
    """
    out = {}
    for p, v in combo.items():
        lo, hi = PARAM_LIMITS[p]
        v = max(lo, min(hi, float(v)))
        if p in ("smoothing_k", "close_k"):
            v_int = int(round(v))
            if v_int < 1:
                v_int = 1
            if v_int % 2 == 0:
                v_lo = max(1, v_int - 1)
                v_hi = v_int + 1
                v_int = v_lo if abs(v - v_lo) <= abs(v - v_hi) else v_hi
            out[p] = float(v_int)
        elif p == "window_pad":
            out[p] = float(int(round(v)))
        else:
            out[p] = round(v, 1)
    return out


def combos_equal(a: dict, b: dict) -> bool:
    """Return True if two clamped combos map to the same physical values."""
    ca = clamp_and_round(a)
    cb = clamp_and_round(b)
    return all(ca[p] == cb[p] for p in SWEPT_PARAMS)


# =============================================================================
# STEP 0 — LOAD PART 1 CORRELATION DATA
# =============================================================================

def load_correlation_data(sensitivity_dir: str) -> tuple[dict, dict]:
    """
    Load correlation_matrix.csv and param_relationship_matrix.csv from Part 1.

    Returns
    -------
    pearson_r : dict  {param: float}   — Pearson r with CER (negative = good)
    rel_matrix: dict  {param: {param: float}}  — scaling factors
    """
    pearson_r  : dict[str, float]             = {}
    rel_matrix : dict[str, dict[str, float]]  = {}

    corr_path = os.path.join(sensitivity_dir, "correlation_matrix.csv")
    if os.path.exists(corr_path):
        with open(corr_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                param = row.get("Parameter", "").strip()
                if param in SWEPT_PARAMS:
                    try:
                        pearson_r[param] = float(row["Pearson_r_with_CER"])
                    except (ValueError, KeyError):
                        pass
        print(f"  [Config] Loaded correlation data: {corr_path}")
    else:
        print(f"  [Config] WARNING: correlation_matrix.csv not found at {corr_path}")
        print(f"  [Config] Using default influence order.")

    rel_path = os.path.join(sensitivity_dir, "param_relationship_matrix.csv")
    if os.path.exists(rel_path):
        with open(rel_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                param_a = row.get("param", "").strip()
                if param_a in SWEPT_PARAMS:
                    rel_matrix[param_a] = {}
                    for param_b in SWEPT_PARAMS:
                        try:
                            rel_matrix[param_a][param_b] = float(row.get(param_b, 0.0))
                        except (ValueError, TypeError):
                            rel_matrix[param_a][param_b] = 0.0
        print(f"  [Config] Loaded relationship matrix: {rel_path}")
    else:
        print(f"  [Config] WARNING: param_relationship_matrix.csv not found at {rel_path}")

    return pearson_r, rel_matrix


def _build_search_config(pearson_r: dict, rel_matrix: dict) -> dict:
    """
    Derive the runtime search configuration from Part 1 correlation data.

    Returns a dict with:
      influence_order   : list of params ordered by |r| descending
      initial_steps     : dict of starting step sizes per param
      diagonal_pairs    : list of (param_a, param_b) pairs to move together
                          (determined by high relationship score ≥ 0.7)
      tiebreak_param    : param with most negative r (prefer higher value)
    """
    config = {
        "influence_order":  DEFAULT_INFLUENCE_ORDER[:],
        "initial_steps":    DEFAULT_INITIAL_STEPS.copy(),
        "diagonal_pairs":   [],
        "tiebreak_param":   "multi_seg_threshold",
    }

    if not pearson_r:
        return config

    # Sort by |r| descending → influence order
    config["influence_order"] = sorted(
        SWEPT_PARAMS, key=lambda p: abs(pearson_r.get(p, 0.0)), reverse=True
    )

    # Most negative r → decreasing CER when param increases → prefer high
    most_neg = min(pearson_r, key=lambda p: pearson_r.get(p, 0.0), default="multi_seg_threshold")
    config["tiebreak_param"] = most_neg

    # Step sizes: inversely proportional to |r| (high influence → finer step)
    abs_r_vals = {p: abs(pearson_r.get(p, 0.001)) for p in SWEPT_PARAMS}
    max_r       = max(abs_r_vals.values()) if abs_r_vals else 1.0
    if max_r > 0:
        for p in SWEPT_PARAMS:
            ratio = abs_r_vals[p] / max_r
            if ratio > 0.5:
                # High influence: fine step
                config["initial_steps"][p] = DEFAULT_INITIAL_STEPS.get(p, 1.0)
            elif ratio > 0.1:
                # Medium: use default
                config["initial_steps"][p] = DEFAULT_INITIAL_STEPS.get(p, 2.0)
            else:
                # Low influence: coarse step to not waste budget
                config["initial_steps"][p] = DEFAULT_INITIAL_STEPS.get(p, 2.0) * 1.5

    # Diagonal pairs: find param pairs with |relationship score| ≥ 0.7
    # (excluding self-pairs and the tiebreak param, which is handled separately)
    if rel_matrix:
        seen = set()
        for a in SWEPT_PARAMS:
            for b in SWEPT_PARAMS:
                if a == b:
                    continue
                key = tuple(sorted([a, b]))
                if key in seen:
                    continue
                score = abs(rel_matrix.get(a, {}).get(b, 0.0))
                if score >= 0.7:
                    config["diagonal_pairs"].append((a, b))
                    seen.add(key)

    print(f"\n  [Config] Search configuration derived from Part 1 data:")
    print(f"    Influence order  : {config['influence_order']}")
    print(f"    Initial steps    : {config['initial_steps']}")
    print(f"    Diagonal pairs   : {config['diagonal_pairs']}")
    print(f"    Tie-break param  : {config['tiebreak_param']}")

    return config


# =============================================================================
# STEP 1 — LOAD PART 1 PER-IMAGE CER TABLE + GLOBAL CENSUS
# =============================================================================

def load_per_image_cer(csv_path: str) -> dict[str, dict[tuple, float]]:
    result: dict[str, dict[tuple, float]] = defaultdict(dict)
    if not os.path.exists(csv_path):
        return result
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row.get("stem", "").strip()
            if not stem:
                continue
            try:
                key = tuple(float(row[p]) for p in SWEPT_PARAMS)
                cer = float(row.get("cer", 100.0))
            except (ValueError, KeyError):
                continue
            result[stem][key] = cer
    return result


def build_global_census(cer_table: dict[str, dict[tuple, float]]) -> dict[tuple, int]:
    """
    For every grid combo, count how many images achieved their personal
    minimum CER at that combo.  Ties within an image are counted for ALL
    tied combos (gives a fair count of "globally robust" points).

    Returns dict: {combo_key_tuple: frequency_count}
    """
    census: dict[tuple, int] = defaultdict(int)
    for stem, stem_table in cer_table.items():
        if not stem_table:
            continue
        min_cer = min(stem_table.values())
        for key, cer in stem_table.items():
            if cer <= min_cer + 1e-9:   # within floating-point tolerance
                census[key] += 1
    return census


# =============================================================================
# STEP 2 — IMAGE FEATURE EXTRACTION
# =============================================================================

def extract_image_features(img_path: str) -> Optional[dict]:
    raw = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    blur       = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary  = cv2.threshold(blur, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    ink_mask = (binary > 0)

    pixel_density        = float(ink_mask.sum()) / (h * w)
    foreground_fraction  = float((gray < 128).sum()) / (h * w)
    bg_pixels            = gray[~ink_mask]
    noise_coefficient    = float(bg_pixels.std()) if len(bg_pixels) > 0 else 0.0

    dist              = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ink_dists         = dist[ink_mask]
    mean_stroke_width = float(ink_dists.mean() * 2) if len(ink_dists) > 0 else 1.0
    aspect_ratio      = float(w) / float(h) if h > 0 else 1.0

    h_proj = ink_mask.sum(axis=0).astype(float)
    v_proj = ink_mask.sum(axis=1).astype(float)
    horizontal_projection_variance = float(h_proj.var())
    vertical_projection_variance   = float(v_proj.var())

    edges        = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).sum()) / (h * w)

    resized_32     = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    local_contrast = float(resized_32.astype(float).std())

    n_components, _ = cv2.connectedComponents(binary)
    n_components    = max(0, int(n_components) - 1)

    projection = ink_mask.sum(axis=0).astype(int)
    gap_lengths: list[int] = []
    in_gap    = False
    gap_start = 0
    for x, val in enumerate(projection):
        if val == 0 and not in_gap:
            gap_start = x
            in_gap    = True
        elif val > 0 and in_gap:
            gap_lengths.append(x - gap_start)
            in_gap = False
    mean_gap_px = float(np.mean(gap_lengths)) if gap_lengths else 0.0
    std_gap_px  = float(np.std(gap_lengths))  if gap_lengths else 0.0

    return {
        "pixel_density":                   round(pixel_density, 6),
        "foreground_fraction":             round(foreground_fraction, 6),
        "noise_coefficient":               round(noise_coefficient, 4),
        "mean_stroke_width":               round(mean_stroke_width, 4),
        "aspect_ratio":                    round(aspect_ratio, 4),
        "horizontal_projection_variance":  round(horizontal_projection_variance, 4),
        "vertical_projection_variance":    round(vertical_projection_variance, 4),
        "edge_density":                    round(edge_density, 6),
        "local_contrast":                  round(local_contrast, 4),
        "n_components":                    n_components,
        "mean_gap_px":                     round(mean_gap_px, 4),
        "std_gap_px":                      round(std_gap_px, 4),
        "image_height_px":                 h,
        "image_width_px":                  w,
    }


FEATURE_KEYS = [
    "pixel_density", "foreground_fraction", "noise_coefficient",
    "mean_stroke_width", "aspect_ratio", "horizontal_projection_variance",
    "vertical_projection_variance", "edge_density", "local_contrast",
    "n_components", "mean_gap_px", "std_gap_px",
    "image_height_px", "image_width_px",
]


# =============================================================================
# STEP 3 — RUN PIPELINE FOR AN OFF-GRID COMBO
# =============================================================================

def run_combo_on_image(
    img_path:     str,
    combo:        dict,
    model:        object,
    idx_to_class: dict,
    base_cfg:     PipelineConfig,
    ground_truth: str,
    run_idx:      int = 0,
) -> Optional[float]:
    """
    Build a PipelineConfig from *combo* (which may have off-grid float values),
    run the OCR pipeline on one image, and return the real CER (0–100).
    """
    import tempfile
    import shutil as _shutil

    fname = os.path.basename(img_path)
    stem  = os.path.splitext(fname)[0]

    run_work = os.path.join(base_cfg.work_root, "phase2_probes",
                            f"probe_{run_idx:06d}")

    img_tmp_dir    = os.path.join(run_work, "img_input")
    label_tmp_path = os.path.join(run_work, "labels.csv")
    os.makedirs(img_tmp_dir, exist_ok=True)

    try:
        dst_img = os.path.join(img_tmp_dir, fname)
        if not os.path.exists(dst_img):
            try:
                os.symlink(os.path.abspath(img_path), dst_img)
            except (OSError, NotImplementedError):
                _shutil.copy2(img_path, dst_img)

        with open(label_tmp_path, "w", encoding="utf-8-sig", newline="") as f:
            gt_escaped = ground_truth.replace('"', '""')
            f.write(f'{stem},"{gt_escaped}"\n')

        cfg = PipelineConfig(
            input_folder        = img_tmp_dir,
            label_csv           = label_tmp_path,
            model_path          = base_cfg.model_path,
            class_map           = base_cfg.class_map,
            work_root           = run_work,
            word_spacer_enabled = base_cfg.word_spacer_enabled,
            fullset             = True,
            sample              = 1,
            seed                = 0,
            target_height       = FIXED_PARAMS["target_height"],
            smoothing_k         = int(combo["smoothing_k"]),
            close_k             = int(combo["close_k"]),
            skeleton_dil        = FIXED_PARAMS["skeleton_dil"],
            valley_min_width    = FIXED_PARAMS["valley_min_width"],
            window_pad          = int(combo["window_pad"]),
            multi_seg_threshold = float(combo["multi_seg_threshold"]),
            word_gap_px         = FIXED_PARAMS["word_gap_px"],
        )

        results = run_pipeline(
            cfg          = cfg,
            model        = model,
            idx_to_class = idx_to_class,
            skip_stage3  = True,
            silent       = True,
        )

        matched = next((r for r in results if r.get("stem") == stem), None)
        if matched is None:
            matched = results[0] if results else {}

        cer = matched.get("cer")
        return float(cer) if cer is not None else None

    except Exception as exc:
        print(f"      [run_combo] ERROR: {exc}")
        return None


# =============================================================================
# STEP 4 — PHASE A: GLOBAL-CENSUS ANCHOR SELECTION
# =============================================================================

def select_anchor(
    stem:           str,
    stem_table:     dict[tuple, float],
    global_census:  dict[tuple, int],
    tiebreak_param: str,
) -> Optional[dict]:
    """
    Phase A: Select the single best-ranked anchor for this image.

    Ranking logic (applied in order):
      1. Minimum CER for this image.
      2. Highest global census frequency (most robust across all images).
      3. Highest value of tiebreak_param (e.g., multi_seg_threshold).
      4. Lexicographic stability (deterministic tie-break of last resort).

    Returns a plain dict keyed by SWEPT_PARAMS, or None if table is empty.
    """
    if not stem_table:
        return None

    min_cer = min(stem_table.values())
    # All combos that achieve (or are within floating-point eps of) the best CER
    candidates = [
        key for key, cer in stem_table.items()
        if cer <= min_cer + 1e-9
    ]

    if len(candidates) == 1:
        return dict(zip(SWEPT_PARAMS, candidates[0]))

    # Sort by: frequency DESC, tiebreak_param value DESC, then tuple for stability
    tb_idx = SWEPT_PARAMS.index(tiebreak_param) if tiebreak_param in SWEPT_PARAMS else -1

    def sort_key(key):
        freq  = global_census.get(key, 0)
        tb    = key[tb_idx] if tb_idx >= 0 else 0.0
        return (-freq, -tb, key)   # negate for descending sort

    candidates.sort(key=sort_key)
    return dict(zip(SWEPT_PARAMS, candidates[0]))


# =============================================================================
# STEP 5 — PHASE B+C: WEIGHTED COORDINATE DESCENT WITH STEP DECAY
# =============================================================================

def _log_hop(writer, fh,
             stem: str, hop: int, phase: str,
             combo: dict, cer: Optional[float],
             accepted: bool, note: str) -> None:
    row = {
        "stem":                stem,
        "hop_number":          hop,
        "phase":               phase,
        "smoothing_k":         combo.get("smoothing_k", ""),
        "close_k":             combo.get("close_k", ""),
        "window_pad":          combo.get("window_pad", ""),
        "multi_seg_threshold": combo.get("multi_seg_threshold", ""),
        "cer":                 round(cer, 4) if cer is not None else "",
        "accepted":            int(accepted),
        "note":                note,
    }
    writer.writerow(row)
    fh.flush()


def _generate_probes(
    center:          dict,
    steps:           dict,
    influence_order: list,
    diagonal_pairs:  list,
) -> list[tuple[dict, str]]:
    """
    Generate a list of (probe_combo, description) tuples to evaluate.

    Probe strategy:
      1. For each param in influence_order, generate +step and -step probes.
      2. For each diagonal pair (a, b), generate a combined move where both
         params move simultaneously in the same direction.
         Only the primary param (first in influence order) drives the step;
         the secondary scales by the ratio of their steps.
      3. Duplicates (after clamping) are filtered out.
    """
    probes    : list[tuple[dict, str]] = []
    seen_keys : set[tuple]             = set()

    def _add(candidate: dict, desc: str):
        clamped = clamp_and_round(candidate)
        key     = combo_key(clamped)
        if key not in seen_keys and not combos_equal(clamped, center):
            seen_keys.add(key)
            probes.append((clamped, desc))

    # Build a set of diagonal pairs for fast lookup
    diagonal_set: set[tuple] = set()
    for a, b in diagonal_pairs:
        diagonal_set.add((a, b))
        diagonal_set.add((b, a))

    already_diagonal: set[str] = set()   # params already covered diagonally

    # ── 1. Diagonal pair probes ───────────────────────────────────────────────
    for a, b in diagonal_pairs:
        step_a = steps.get(a, 1.0)
        step_b = steps.get(b, 1.0)
        for sign in (+1, -1):
            candidate = center.copy()
            candidate[a] = center[a] + sign * step_a
            candidate[b] = center[b] + sign * step_b
            direction = "+" if sign > 0 else "-"
            _add(candidate, f"diagonal {direction} ({a},{b})")
        already_diagonal.add(a)
        already_diagonal.add(b)

    # ── 2. Coordinate probes for remaining (non-diagonal) params ─────────────
    for param in influence_order:
        step = steps.get(param, 1.0)
        for sign in (+1, -1):
            candidate = center.copy()
            candidate[param] = center[param] + sign * step
            direction = "+" if sign > 0 else "-"
            _add(candidate, f"coord {direction} {param}")

    return probes


def pattern_search_image(
    stem:            str,
    img_path:        str,
    cer_table:       dict[str, dict[tuple, float]],
    global_census:   dict[tuple, int],
    model:           object,
    idx_to_class:    dict,
    base_cfg:        PipelineConfig,
    ground_truth:    str,
    max_hops:        int,
    min_cer_delta:   float,
    search_config:   dict,
    log_writer,
    log_fh,
) -> tuple[dict, float, int]:
    """
    Three-phase pattern search for one image.

    Phase A  (free): Census-ranked anchor selection.
    Phase B  (≤ max_hops runs): Weighted coordinate probing.
    Phase C  (within B budget): Step decay for precision convergence.

    Returns (best_combo_dict, best_cer, pipeline_runs_used).
    """
    stem_table = cer_table.get(stem, {})
    if not stem_table:
        fallback = {p: SWEPT_GRID[p][len(SWEPT_GRID[p]) // 2] for p in SWEPT_PARAMS}
        return fallback, 100.0, 0

    influence_order = search_config["influence_order"]
    initial_steps   = search_config["initial_steps"].copy()
    diagonal_pairs  = search_config["diagonal_pairs"]
    tiebreak_param  = search_config["tiebreak_param"]

    # ── Phase A ───────────────────────────────────────────────────────────────
    anchor = select_anchor(stem, stem_table, global_census, tiebreak_param)
    if anchor is None:
        fallback = {p: SWEPT_GRID[p][len(SWEPT_GRID[p]) // 2] for p in SWEPT_PARAMS}
        return fallback, 100.0, 0

    anchor_cer = stem_table[combo_key(anchor)]
    best_combo = anchor.copy()
    best_cer   = anchor_cer

    print(f"    [A] Anchor (census-ranked): {anchor}  CER={anchor_cer:.2f}%")
    _log_hop(log_writer, log_fh, stem, 0, "A",
             anchor, anchor_cer, True,
             f"phase-A census anchor  grid_combos={len(stem_table)}")

    # ── Phase B+C — decaying coordinate descent ───────────────────────────────
    steps         = initial_steps.copy()
    pipeline_runs = 0
    hop_num       = 1
    sweep_count   = 0
    _probe_base   = abs(hash(stem)) % 10_000_000

    # Convergence: stop when threshold step is tiny
    conv_param = tiebreak_param   # most influential param drives convergence

    while pipeline_runs < max_hops:
        # Generate all probes for this sweep
        probes = _generate_probes(
            center          = best_combo,
            steps           = steps,
            influence_order = influence_order,
            diagonal_pairs  = diagonal_pairs,
        )

        if not probes:
            print(f"    [B] No distinct probes — converged.")
            break

        sweep_improved = False
        sweep_count   += 1

        for probe_combo, probe_desc in probes:
            if pipeline_runs >= max_hops:
                break

            cer = run_combo_on_image(
                img_path     = img_path,
                combo        = probe_combo,
                model        = model,
                idx_to_class = idx_to_class,
                base_cfg     = base_cfg,
                ground_truth = ground_truth,
                run_idx      = _probe_base + pipeline_runs,
            )
            pipeline_runs += 1

            if cer is None:
                _log_hop(log_writer, log_fh, stem, hop_num, "B",
                         probe_combo, None, False,
                         f"pipeline error  [{probe_desc}]")
                hop_num += 1
                continue

            improvement = best_cer - cer
            accepted    = improvement > min_cer_delta

            note = (
                f"NEW BEST  Δ={improvement:.2f}%  sweep={sweep_count}  [{probe_desc}]"
                if accepted
                else f"no gain ({cer:.2f}% vs {best_cer:.2f}%)  [{probe_desc}]"
            )

            print(f"    [B] hop {hop_num}: {probe_desc}  "
                  f"{probe_combo}  CER={cer:.2f}%  "
                  f"{'✓ NEW BEST' if accepted else '✗'}")
            _log_hop(log_writer, log_fh, stem, hop_num, "B",
                     probe_combo, cer, accepted, note)
            hop_num += 1

            if accepted:
                best_combo     = probe_combo.copy()
                best_cer       = cer
                sweep_improved = True
                # After finding a new best in this sweep, continue checking
                # remaining probes in the same sweep (they may be even better)

        # ── Phase C — step decay if no improvement in full sweep ─────────────
        if not sweep_improved:
            # Halve all step sizes
            steps = {p: v * 0.5 for p, v in steps.items()}
            print(f"    [C] Sweep {sweep_count} — no gain. "
                  f"Step decay → {conv_param}_step={steps[conv_param]:.3f}")
            _log_hop(log_writer, log_fh, stem, hop_num, "C",
                     best_combo, best_cer, False,
                     f"step decay sweep={sweep_count}  "
                     f"{conv_param}_step={steps[conv_param]:.3f}")
            hop_num += 1

            # Convergence check
            if steps[conv_param] < CONVERGENCE_STEP_THRESHOLD:
                print(f"    [C] Converged — {conv_param} step "
                      f"< {CONVERGENCE_STEP_THRESHOLD}")
                break

    return best_combo, best_cer, pipeline_runs


# =============================================================================
# STEP 6 — TRAIN MULTI-OUTPUT RANDOM FOREST
# =============================================================================

def train_multioutput_rf(records: list[dict],
                          out_dir: str,
                          test_split: float = TEST_SPLIT) -> None:
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        print(f"  ERROR: scikit-learn or joblib not installed: {e}")
        print("  Run: pip install scikit-learn joblib")
        sys.exit(1)

    valid = [r for r in records if r.get("best_cer", 999) < 999]
    if len(valid) < 4:
        print(f"  WARNING: Only {len(valid)} valid records — cannot train reliably.")
        return

    X = np.array([[r[k] for k in FEATURE_KEYS] for r in valid], dtype=float)
    Y = np.array([
        [r["smoothing_k"], r["close_k"], r["window_pad"], r["multi_seg_threshold"]]
        for r in valid
    ], dtype=float)

    print(f"\n  [{_ts()}] Training on {len(valid)} samples, "
          f"{len(FEATURE_KEYS)} features, {len(SWEPT_PARAMS)} output targets")

    if len(valid) >= 6:
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            X, Y, test_size=test_split, random_state=42
        )
    else:
        X_tr, X_te, Y_tr, Y_te = X, X, Y, Y

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    base_rf = RandomForestClassifier(
        n_estimators      = 100,
        max_depth         = 8,
        min_samples_split = 3,
        random_state      = 42,
        n_jobs            = S1_WORKERS,
    )
    clf = MultiOutputClassifier(base_rf, n_jobs=S1_WORKERS)
    clf.fit(X_tr_s, Y_tr)
    Y_pred = clf.predict(X_te_s)

    per_acc = []
    for i, p in enumerate(SWEPT_PARAMS):
        acc = accuracy_score(Y_te[:, i], Y_pred[:, i])
        per_acc.append(acc)
        print(f"    {p:<28} accuracy: {acc*100:.1f}%")

    mean_acc = float(np.mean(per_acc))
    print(f"  Mean per-output accuracy : {mean_acc*100:.1f}%")

    rf_path     = os.path.join(out_dir, "random_forest_model.joblib")
    scaler_path = os.path.join(out_dir, "feature_scaler.joblib")
    joblib.dump(clf,    rf_path)
    joblib.dump(scaler, scaler_path)
    print(f"  random_forest_model.joblib → {rf_path}")
    print(f"  feature_scaler.joblib      → {scaler_path}")

    report_lines = [
        "=" * 62,
        "  PART 2 — Multi-Output RF Validation Report",
        f"  Generated : {_ts()}",
        "=" * 62,
        "",
        f"  Training samples : {len(X_tr)}",
        f"  Test samples     : {len(X_te)}",
        f"  Features         : {len(FEATURE_KEYS)}",
        f"  Output targets   : {len(SWEPT_PARAMS)}",
        "",
        "  ── Per-output accuracy ─────────────────────────────────",
    ]
    for p, acc in zip(SWEPT_PARAMS, per_acc):
        bar = "▮" * int(acc * 40)
        report_lines.append(f"    {p:<28} {acc*100:5.1f}%  {bar}")
    report_lines += [
        "",
        f"  Mean accuracy : {mean_acc*100:.1f}%",
        "",
        "  ── Feature importances (averaged across estimators) ────",
    ]
    try:
        imp_matrix = np.array([
            est.feature_importances_
            for est in clf.estimators_
        ])
        mean_imp   = imp_matrix.mean(axis=0)
        sorted_imp = sorted(zip(FEATURE_KEYS, mean_imp),
                            key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_imp[:10]:
            bar = "▮" * int(imp * 40)
            report_lines.append(f"    {feat:<38} {imp:.4f}  {bar}")
    except Exception:
        pass

    report_lines += ["", "=" * 62]
    report_path = os.path.join(out_dir, "model_validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"  model_validation_report.txt → {report_path}")


# =============================================================================
# MAIN ENTRY
# =============================================================================

def run_heuristic(sensitivity_dir: str   = SENSITIVITY_DIR,
                   out_dir:         str   = PHASE2_OUT,
                   sample:          int   = MAPPING_SAMPLE,
                   seed:            int   = MAPPING_SEED,
                   max_hops:        int   = MAX_HOPS,
                   min_cer_delta:   float = MIN_CER_DELTA,
                   input_folder:    str   = INPUT_FOLDER,
                   label_csv:       str   = LABEL_CSV,
                   model_path:      str   = MODEL_PATH,
                   class_map:       str   = CLASS_MAP,
                   base_work_root:  str   = WORK_ROOT,
                   word_spacer:     bool  = True) -> None:

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    print(f"\n{'='*65}")
    print(f"  Part 2 — Statistical Pattern Search & Multi-Output RF Training")
    print(f"  Started : {_ts()}")
    print(f"  Sensitivity dir : {sensitivity_dir}")
    print(f"  Output dir      : {out_dir}")
    print(f"  Sample          : {sample}  (seed={seed})")
    print(f"  Max pipeline hops per image : {max_hops}")
    print(f"  Min CER delta (noise floor) : {min_cer_delta}%")
    print(f"{'='*65}\n")

    # ── Verify Part 1 outputs ─────────────────────────────────────────────────
    per_image_csv   = os.path.join(sensitivity_dir, "per_image_cer.csv")
    rel_matrix_csv  = os.path.join(sensitivity_dir, "param_relationship_matrix.csv")
    directions_json = os.path.join(sensitivity_dir, "param_directions.json")
    corr_csv        = os.path.join(sensitivity_dir, "correlation_matrix.csv")

    for path, label in [
        (per_image_csv,   "per_image_cer.csv"),
        (rel_matrix_csv,  "param_relationship_matrix.csv"),
        (directions_json, "param_directions.json"),
        (corr_csv,        "correlation_matrix.csv"),
    ]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} not found at: {path}")
            print("  Run part1_sensitivity.py first.")
            sys.exit(1)

    # ── Load correlation data and build search config ─────────────────────────
    pearson_r, rel_matrix = load_correlation_data(sensitivity_dir)
    search_config         = _build_search_config(pearson_r, rel_matrix)

    # ── Save the derived search config for reproducibility ───────────────────
    search_config_path = os.path.join(out_dir, "search_config.json")
    with open(search_config_path, "w", encoding="utf-8") as f:
        json.dump(search_config, f, ensure_ascii=False, indent=2)
    print(f"\n  search_config.json → {search_config_path}")

    # ── Load CER table ────────────────────────────────────────────────────────
    print(f"\n  [{_ts()}] Loading Part 1 CER table …")
    cer_table = load_per_image_cer(per_image_csv)
    print(f"  CER table: {sum(len(v) for v in cer_table.values())} rows "
          f"across {len(cer_table)} images")

    # ── Build global census ───────────────────────────────────────────────────
    print(f"  [{_ts()}] Building global grid census …")
    global_census = build_global_census(cer_table)
    top_census    = sorted(global_census.items(), key=lambda kv: kv[1], reverse=True)
    print(f"  Global census: {len(global_census)} distinct grid points")
    if top_census:
        top_key, top_freq = top_census[0]
        top_combo         = dict(zip(SWEPT_PARAMS, top_key))
        print(f"  Most robust grid point: {top_combo}  (best for {top_freq} images)")

    # Save census as CSV for inspection
    census_path = os.path.join(out_dir, "grid_popularity_rank.csv")
    with open(census_path, "w", newline="", encoding="utf-8-sig") as cf:
        writer = csv.writer(cf)
        writer.writerow(SWEPT_PARAMS + ["frequency"])
        for key, freq in sorted(global_census.items(), key=lambda kv: kv[1], reverse=True):
            writer.writerow(list(key) + [freq])
    print(f"  grid_popularity_rank.csv → {census_path}")

    # ── Load model once ───────────────────────────────────────────────────────
    print(f"\n  [{_ts()}] Loading model …")
    with open(class_map, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    model = _load_model(model_path, len(idx_to_class), _DEVICE)
    print(f"  Model loaded on {_DEVICE}\n")

    # ── Load labels ───────────────────────────────────────────────────────────
    labels = _load_labels(label_csv)

    # ── Base PipelineConfig (paths only) ──────────────────────────────────────
    base_cfg = PipelineConfig(
        input_folder        = input_folder,
        label_csv           = label_csv,
        model_path          = model_path,
        class_map           = class_map,
        work_root           = base_work_root,
        word_spacer_enabled = word_spacer,
    )

    # ── Select the same image set as Part 1 ───────────────────────────────────
    valid_exts = {".png", ".jpg", ".jpeg"}
    all_files  = sorted([
        os.path.join(input_folder, fn)
        for fn in os.listdir(input_folder)
        if os.path.splitext(fn.lower())[1] in valid_exts
    ])
    _random.seed(seed)
    img_paths = _random.sample(all_files, min(sample, len(all_files)))
    print(f"  [{_ts()}] Selected {len(img_paths)} images (seed={seed})\n")

    # ── Open live CSV writers ─────────────────────────────────────────────────
    hop_log_fields = [
        "stem", "hop_number", "phase",
        "smoothing_k", "close_k", "window_pad", "multi_seg_threshold",
        "cer", "accepted", "note",
    ]
    stats_fields = (
        ["stem"] + FEATURE_KEYS +
        ["smoothing_k", "close_k", "window_pad", "multi_seg_threshold",
         "best_cer", "pipeline_runs", "grid_best_cer", "continuous_gain"]
    )
    summary_fields = [
        "stem", "grid_best_cer", "best_cer", "continuous_gain",
        "pipeline_runs",
        "smoothing_k", "close_k", "window_pad", "multi_seg_threshold",
    ]

    hop_fh,   hop_writer   = _open_csv_appender(
        os.path.join(out_dir, "hill_climb_log.csv"), hop_log_fields)
    stats_fh, stats_writer = _open_csv_appender(
        os.path.join(out_dir, "image_stats_with_best_combo.csv"), stats_fields)
    summ_fh,  summ_writer  = _open_csv_appender(
        os.path.join(out_dir, "climb_summary.csv"), summary_fields)

    all_records = []

    # ── Process each image ────────────────────────────────────────────────────
    for img_idx, img_path in enumerate(img_paths, 1):
        fname = os.path.basename(img_path)
        stem  = os.path.splitext(fname)[0]

        print(f"  [{_ts()}] [{img_idx}/{len(img_paths)}] {fname}")

        stem_table = cer_table.get(stem, {})
        if not stem_table:
            print(f"    SKIP — no CER data in per_image_cer.csv for this stem")
            continue

        grid_best_cer = min(stem_table.values())

        features = extract_image_features(img_path)
        if features is None:
            print(f"    SKIP — could not decode image")
            continue

        ground_truth = labels.get(stem, "")

        best_combo, best_cer, pipeline_runs = pattern_search_image(
            stem          = stem,
            img_path      = img_path,
            cer_table     = cer_table,
            global_census = global_census,
            model         = model,
            idx_to_class  = idx_to_class,
            base_cfg      = base_cfg,
            ground_truth  = ground_truth,
            max_hops      = max_hops,
            min_cer_delta = min_cer_delta,
            search_config = search_config,
            log_writer    = hop_writer,
            log_fh        = hop_fh,
        )

        continuous_gain = grid_best_cer - best_cer
        print(f"    Final : {best_combo}  CER={best_cer:.2f}%  "
              f"grid_best={grid_best_cer:.2f}%  gain={continuous_gain:.2f}%  "
              f"runs={pipeline_runs}")

        # ── Write to live CSVs ────────────────────────────────────────────────
        stats_row = {
            "stem": stem, **features, **best_combo,
            "best_cer":        round(best_cer, 4),
            "pipeline_runs":   pipeline_runs,
            "grid_best_cer":   round(grid_best_cer, 4),
            "continuous_gain": round(continuous_gain, 4),
        }
        stats_writer.writerow(stats_row)
        stats_fh.flush()

        summ_row = {
            "stem":            stem,
            "grid_best_cer":   round(grid_best_cer, 4),
            "best_cer":        round(best_cer, 4),
            "continuous_gain": round(continuous_gain, 4),
            "pipeline_runs":   pipeline_runs,
            **best_combo,
        }
        summ_writer.writerow(summ_row)
        summ_fh.flush()

        all_records.append({
            "stem": stem, **features, **best_combo,
            "best_cer":        round(best_cer, 4),
            "pipeline_runs":   pipeline_runs,
            "grid_best_cer":   round(grid_best_cer, 4),
            "continuous_gain": round(continuous_gain, 4),
        })

    # ── Close writers ─────────────────────────────────────────────────────────
    hop_fh.close()
    stats_fh.close()
    summ_fh.close()

    print(f"\n  [{_ts()}] Search complete for {len(all_records)} images.")

    if not all_records:
        print("  WARNING: No records produced — check per_image_cer.csv stems "
              "match image filenames.")
        return

    # ── Summary stats ─────────────────────────────────────────────────────────
    gains     = [r["continuous_gain"] for r in all_records]
    improved  = sum(1 for g in gains if g > min_cer_delta)
    mean_gain = float(np.mean(gains)) if gains else 0.0
    print(f"  Images improved beyond grid best : {improved}/{len(all_records)}")
    print(f"  Mean CER gain from grid best     : {mean_gain:.3f}%")

    # ── Train RF ──────────────────────────────────────────────────────────────
    print(f"\n  [{_ts()}] Training Multi-Output Random Forest …")
    train_multioutput_rf(all_records, out_dir)

    # ── Save param_profiles.json ──────────────────────────────────────────────
    seen_combos: set[tuple] = set()
    profiles: list[dict]   = []
    for r in sorted(all_records, key=lambda x: x["best_cer"]):
        key = combo_key({p: r[p] for p in SWEPT_PARAMS})
        if key not in seen_combos:
            seen_combos.add(key)
            profiles.append({p: r[p] for p in SWEPT_PARAMS})
    profiles_path = os.path.join(out_dir, "param_profiles.json")
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    print(f"  param_profiles.json        → {profiles_path}")

    elapsed = time.time() - t0
    print(f"\n  [{_ts()}] Part 2 complete in {elapsed:.1f}s")
    print(f"  Artefacts saved to: {out_dir}")
    print("=" * 65 + "\n")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Part 2: Statistical Pattern Search & multi-output RF training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--sensitivity_dir", default=SENSITIVITY_DIR)
    ap.add_argument("--out_dir",         default=PHASE2_OUT)
    ap.add_argument("--sample",          type=int,   default=MAPPING_SAMPLE,
                    help="Must match SENS_SAMPLE in Part 1")
    ap.add_argument("--seed",            type=int,   default=MAPPING_SEED,
                    help="Must match SENS_SEED in Part 1")
    ap.add_argument("--max_hops",        type=int,   default=MAX_HOPS,
                    help="Max pipeline runs per image")
    ap.add_argument("--min_cer_delta",   type=float, default=MIN_CER_DELTA,
                    help="Minimum CER improvement to count as gain")
    ap.add_argument("--input_folder",    default=INPUT_FOLDER)
    ap.add_argument("--label_csv",       default=LABEL_CSV)
    ap.add_argument("--model_path",      default=MODEL_PATH)
    ap.add_argument("--class_map",       default=CLASS_MAP)
    ap.add_argument("--work_root",       default=WORK_ROOT)
    ap.add_argument("--no_word_spacer",  action="store_true")
    args = ap.parse_args()

    run_heuristic(
        sensitivity_dir = args.sensitivity_dir,
        out_dir         = args.out_dir,
        sample          = args.sample,
        seed            = args.seed,
        max_hops        = args.max_hops,
        min_cer_delta   = args.min_cer_delta,
        input_folder    = args.input_folder,
        label_csv       = args.label_csv,
        model_path      = args.model_path,
        class_map       = args.class_map,
        base_work_root  = args.work_root,
        word_spacer     = not args.no_word_spacer,
    )