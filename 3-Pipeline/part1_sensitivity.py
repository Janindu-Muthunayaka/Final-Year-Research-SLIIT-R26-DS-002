# =============================================================================
# part1_sensitivity.py  —  Phase 1: Parameter Sensitivity & Grid Search
#
# Purpose
# -------
# Exhaustively sweep the preprocessing parameter space and measure the
# Character Error Rate (CER) produced by each combination.
#
# What it does
# ------------
# 1. Defines a discrete parameter grid over all tunable knobs.
# 2. Selects a representative sample of images (SENS_SAMPLE, adjustable).
# 3. Runs the merged pipeline for every parameter permutation — the model
#    is loaded ONCE and reused.
# 4. Saves per-image CER per combo to per_image_cer.csv (live-appended).
# 5. Computes mean CER and WER per combination.
# 6. Fits a quadratic curve per swept parameter → extracts direction and
#    sensitivity → writes param_directions.json.
# 7. Builds an inter-parameter relationship matrix → writes
#    param_relationship_matrix.csv.
# 8. Keeps all existing outputs (sensitivity_results.csv,
#    correlation_matrix.csv, sensitivity_report.txt, best_params.json).
#
# Usage
# -----
#   python part1_sensitivity.py
#   python part1_sensitivity.py --sample 30 --seed 99
#   python part1_sensitivity.py --quick
# =============================================================================

from __future__ import annotations

import os
import sys
import json
import csv
import time
import argparse
import itertools
from datetime import datetime
from typing import Any

import numpy as np

# ── Make pipeline_core importable from the same folder ───────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pipeline_core import (
    PipelineConfig,
    run_pipeline,
    _load_model,
    _DEVICE,
    INPUT_FOLDER, LABEL_CSV, MODEL_PATH, CLASS_MAP, WORK_ROOT,
)

import torch

# =============================================================================
# TOP-OF-FILE CONFIGURATION  (adjust here)
# =============================================================================

# Output folder for all sensitivity artefacts
SENSITIVITY_OUT = os.path.join(WORK_ROOT, "Info", "phase1_sensitivity")

# ── Sample size and seed — keep consistent with Part 2 ───────────────────────
SENS_SAMPLE = 250   # ← adjust here; must match MAPPING_SAMPLE in part2
SENS_SEED   = 42   # ← adjust here; must match MAPPING_SEED  in part2

# =============================================================================
# PARAMETER GRID
# =============================================================================

FIXED_PARAMS = {
    "target_height":    384,
    "skeleton_dil":     1,
    "valley_min_width": 2,
    "word_gap_px":      35,
}

# FULL GRID — 3 x 3 x 4 x 3 = 108 combinations
FULL_GRID = {
    "smoothing_k":         [1, 3, 5],
    "close_k":             [2, 3, 5],
    "window_pad":          [4, 8, 12, 20],
    "multi_seg_threshold": [95.0, 97.0, 99.0],
}

# QUICK GRID — 2 x 2 x 2 x 2 = 16 combinations (fast sanity check)
QUICK_GRID = {
    "smoothing_k":         [1, 3],
    "close_k":             [2, 3],
    "window_pad":          [8, 12],
    "multi_seg_threshold": [95.0, 97.0],
}

# =============================================================================
# HELPERS
# =============================================================================

def _ts() -> str:
    """Return a short timestamp string for console output."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_combinations(grid: dict) -> list[dict]:
    keys   = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def pearson_r(x: list, y: list) -> float:
    n  = len(x)
    if n < 2:
        return 0.0
    ax = sum(x) / n
    ay = sum(y) / n
    num = sum((xi - ax) * (yi - ay) for xi, yi in zip(x, y))
    den = (sum((xi - ax) ** 2 for xi in x) *
           sum((yi - ay) ** 2 for yi in y)) ** 0.5
    return num / den if den else 0.0


def compute_correlation_matrix(rows: list[dict],
                                param_keys: list[str]) -> dict:
    cer_vals = [r["mean_cer"] for r in rows]
    corr     = {}
    for key in param_keys:
        p_vals    = [r[key] for r in rows]
        corr[key] = round(pearson_r(p_vals, cer_vals), 4)
    return corr


def rank_combinations(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: r["mean_cer"])


# =============================================================================
# QUADRATIC ANALYSIS — direction + sensitivity per parameter
# =============================================================================

def fit_quadratic_direction_sensitivity(
        all_rows: list[dict],
        param_keys: list[str],
        grid: dict) -> dict:
    """
    For each swept parameter, hold all others at their median grid value.
    Collect (param_value, mean_CER) pairs.  Fit a degree-2 polynomial.
    Extract:
      direction   : +1 if increasing the param reduces CER, else -1
      sensitivity : |slope| at the median value (first derivative of quad)

    Returns dict: { param_name: {"direction": int, "sensitivity": float} }
    """
    valid_rows = [r for r in all_rows if r["mean_cer"] < 999.0]
    if not valid_rows:
        return {k: {"direction": 1, "sensitivity": 0.0} for k in param_keys}

    # Median grid index for each parameter
    medians = {k: sorted(grid[k])[len(grid[k]) // 2] for k in param_keys}

    results = {}
    for focal in param_keys:
        other_params = [k for k in param_keys if k != focal]

        # Filter: all other params at their median value
        filtered = [
            r for r in valid_rows
            if all(r[p] == medians[p] for p in other_params)
        ]

        if len(filtered) < 2:
            results[focal] = {"direction": 1, "sensitivity": 0.0}
            continue

        # Aggregate: average CER for each value of focal param
        from collections import defaultdict
        bucket: dict[float, list] = defaultdict(list)
        for r in filtered:
            bucket[float(r[focal])].append(r["mean_cer"])
        xs = sorted(bucket.keys())
        ys = [float(np.mean(bucket[x])) for x in xs]

        if len(xs) < 2:
            results[focal] = {"direction": 1, "sensitivity": 0.0}
            continue

        # Fit quadratic (or linear if only 2 points)
        degree = min(2, len(xs) - 1)
        try:
            coeffs = np.polyfit(xs, ys, degree)   # highest-degree first
        except Exception:
            results[focal] = {"direction": 1, "sensitivity": 0.0}
            continue

        # Evaluate first derivative at median x
        x0 = float(medians[focal])
        if degree == 2:
            a, b, _ = coeffs
            slope_at_x0 = 2 * a * x0 + b
        else:
            # Linear: coeffs = [slope, intercept]
            slope_at_x0 = float(coeffs[0])

        # Negative slope → increasing param reduces CER → direction = +1
        direction   = 1 if slope_at_x0 < 0 else -1
        sensitivity = round(abs(slope_at_x0), 4)

        results[focal] = {"direction": direction, "sensitivity": sensitivity}

    return results


# =============================================================================
# INTER-PARAMETER RELATIONSHIP MATRIX
# =============================================================================

def compute_relationship_matrix(
        all_rows: list[dict],
        param_keys: list[str],
        grid: dict) -> dict[str, dict[str, float]]:
    """
    For every pair (A, B) compute:
      scaling_factor = (dCER/dA) / (dCER/dB)

    Positive  → A and B should move in the same direction.
    Negative  → opposite directions.
    Near zero → independent.

    Returns nested dict: matrix[A][B] = scaling_factor
    Diagonal is always 1.0.
    """
    valid_rows = [r for r in all_rows if r["mean_cer"] < 999.0]
    medians    = {k: sorted(grid[k])[len(grid[k]) // 2] for k in param_keys}

    # Compute slope for every parameter at its median
    slopes: dict[str, float] = {}
    for focal in param_keys:
        other_params = [k for k in param_keys if k != focal]
        filtered = [
            r for r in valid_rows
            if all(r[p] == medians[p] for p in other_params)
        ]
        from collections import defaultdict
        bucket: dict[float, list] = defaultdict(list)
        for r in filtered:
            bucket[float(r[focal])].append(r["mean_cer"])
        xs = sorted(bucket.keys())
        ys = [float(np.mean(bucket[x])) for x in xs]

        if len(xs) < 2:
            slopes[focal] = 0.0
            continue
        degree = min(2, len(xs) - 1)
        try:
            coeffs = np.polyfit(xs, ys, degree)
        except Exception:
            slopes[focal] = 0.0
            continue
        x0 = float(medians[focal])
        if degree == 2:
            a, b, _ = coeffs
            slopes[focal] = 2 * a * x0 + b
        else:
            slopes[focal] = float(coeffs[0])

    # Build matrix
    matrix: dict[str, dict[str, float]] = {}
    for a in param_keys:
        matrix[a] = {}
        for b in param_keys:
            if a == b:
                matrix[a][b] = 1.0
            elif slopes[b] != 0.0:
                ratio = slopes[a] / slopes[b]
                matrix[a][b] = round(ratio, 4)
            else:
                matrix[a][b] = 0.0

    return matrix


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_sensitivity(grid:            dict,
                    fixed:           dict  = None,
                    sample:          int   = SENS_SAMPLE,
                    seed:            int   = SENS_SEED,
                    out_dir:         str   = SENSITIVITY_OUT,
                    input_folder:    str   = INPUT_FOLDER,
                    label_csv:       str   = LABEL_CSV,
                    model_path:      str   = MODEL_PATH,
                    class_map:       str   = CLASS_MAP,
                    base_work_root:  str   = WORK_ROOT,
                    word_spacer:     bool  = True) -> list[dict]:
    """
    Run the pipeline for every parameter combination and collect CER / WER.
    Returns list of combo-level result dicts sorted by mean_cer.
    """
    if fixed is None:
        fixed = FIXED_PARAMS

    os.makedirs(out_dir, exist_ok=True)

    combos = build_combinations(grid)
    total  = len(combos)

    print(f"\n{'='*65}")
    print(f"  Part 1 — Parameter Sensitivity & Grid Search")
    print(f"  Started : {_ts()}")
    print(f"  Swept combinations : {total}")
    print(f"  Fixed params       : {fixed}")
    print(f"  Sample per run     : {sample}  (seed={seed})")
    print(f"  Output             : {out_dir}")
    print(f"{'='*65}\n")

    # ── per_image_cer.csv — opened once, appended throughout ─────────────────
    per_image_csv_path = os.path.join(out_dir, "per_image_cer.csv")
    per_image_fieldnames = (
        list(grid.keys()) +
        ["run_tag", "stem", "cer", "wer"]
    )
    # Write header only if file does not already exist (safe for re-runs)
    write_header = not os.path.exists(per_image_csv_path)
    per_image_fh = open(per_image_csv_path, "a", newline="",
                        encoding="utf-8-sig")
    per_image_writer = csv.DictWriter(
        per_image_fh, fieldnames=per_image_fieldnames, extrasaction="ignore"
    )
    if write_header:
        per_image_writer.writeheader()
        per_image_fh.flush()

    # ── Load model ONCE ───────────────────────────────────────────────────────
    device = _DEVICE
    with open(class_map, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)

    print(f"  [{_ts()}] Loading model from: {model_path}")
    model = _load_model(model_path, num_classes, device)
    print(f"  [{_ts()}] Model loaded.  Starting grid search …\n")

    rows           = []
    all_per_image  = []   # flat list used later for analysis
    total_start    = time.time()
    failed_count   = 0

    for run_idx, combo in enumerate(combos, 1):
        run_tag   = f"run_{run_idx:05d}"
        work_root = os.path.join(base_work_root, "phase1_runs", run_tag)

        cfg = PipelineConfig(
            input_folder        = input_folder,
            label_csv           = label_csv,
            model_path          = model_path,
            class_map           = class_map,
            work_root           = work_root,
            fullset             = False,
            sample              = sample,
            seed                = seed,
            word_spacer_enabled = word_spacer,
            **fixed,
            **combo,
        )

        eta_str = ""
        if run_idx > 1:
            elapsed   = time.time() - total_start
            rate      = elapsed / (run_idx - 1)
            remaining = rate * (total - run_idx + 1)
            eta_str   = f"  ETA ~{remaining/60:.1f}min"

        print(f"  [{_ts()}] [{run_idx}/{total}]{eta_str}  "
              f"sk={combo['smoothing_k']} "
              f"ck={combo['close_k']} "
              f"wp={combo['window_pad']} "
              f"mst={combo['multi_seg_threshold']}")

        try:
            results  = run_pipeline(
                cfg          = cfg,
                model        = model,
                idx_to_class = idx_to_class,
                skip_stage3  = True,
                silent       = True,
            )
            mean_cer = results.mean_cer
            mean_wer = results.mean_wer
            n_valid  = sum(1 for r in results if r.get("ground_truth"))

            # ── Append per-image rows ─────────────────────────────────────────
            for r in results:
                stem = r.get("stem", "")
                cer  = r.get("cer", 0.0)
                wer  = r.get("wer", 0.0)
                pi_row = {**combo,
                          "run_tag": run_tag,
                          "stem":    stem,
                          "cer":     round(cer, 4),
                          "wer":     round(wer, 4)}
                per_image_writer.writerow(pi_row)
                all_per_image.append(pi_row)
            per_image_fh.flush()

            print(f"           → mean CER: {mean_cer:.2f}%  WER: {mean_wer:.2f}%  "
                  f"(GT images: {n_valid})")

        except Exception as exc:
            import traceback
            print(f"           ERROR: {exc}")
            traceback.print_exc()
            mean_cer = 999.0
            mean_wer = 999.0
            n_valid  = 0
            failed_count += 1

        row = {**combo,
               "run_tag":  run_tag,
               "mean_cer": round(mean_cer, 4),
               "mean_wer": round(mean_wer, 4),
               "n_valid":  n_valid}
        rows.append(row)

    per_image_fh.close()

    # ── Sort by CER ───────────────────────────────────────────────────────────
    rows_sorted  = rank_combinations(rows)
    param_keys   = list(grid.keys())
    valid_rows   = [r for r in rows if r["mean_cer"] < 999.0]
    correlations = compute_correlation_matrix(valid_rows, param_keys) if valid_rows else {}
    total_elapsed = time.time() - total_start

    print(f"\n  [{_ts()}] Grid search complete in {total_elapsed:.1f}s")
    print(f"  [{_ts()}] Running quadratic analysis …")

    # ── Quadratic analysis — direction + sensitivity ──────────────────────────
    param_directions = fit_quadratic_direction_sensitivity(
        all_rows   = valid_rows,
        param_keys = param_keys,
        grid       = grid,
    )
    directions_path = os.path.join(out_dir, "param_directions.json")
    with open(directions_path, "w", encoding="utf-8") as f:
        json.dump(param_directions, f, ensure_ascii=False, indent=2)
    print(f"  param_directions.json       → {directions_path}")

    # ── Relationship matrix ───────────────────────────────────────────────────
    print(f"  [{_ts()}] Building relationship matrix …")
    rel_matrix = compute_relationship_matrix(
        all_rows   = valid_rows,
        param_keys = param_keys,
        grid       = grid,
    )
    rel_matrix_path = os.path.join(out_dir, "param_relationship_matrix.csv")
    with open(rel_matrix_path, "w", newline="", encoding="utf-8-sig") as cf:
        writer = csv.writer(cf)
        writer.writerow(["param"] + param_keys)
        for a in param_keys:
            writer.writerow([a] + [rel_matrix[a][b] for b in param_keys])
    print(f"  param_relationship_matrix.csv → {rel_matrix_path}")

    # ── Write main sensitivity CSV ────────────────────────────────────────────
    csv_path   = os.path.join(out_dir, "sensitivity_results.csv")
    fieldnames = param_keys + ["run_tag", "mean_cer", "mean_wer", "n_valid"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    print(f"  sensitivity_results.csv     → {csv_path}")

    # ── Write correlation CSV ─────────────────────────────────────────────────
    corr_csv = os.path.join(out_dir, "correlation_matrix.csv")
    with open(corr_csv, "w", newline="", encoding="utf-8-sig") as cf:
        writer = csv.writer(cf)
        writer.writerow(["Parameter", "Pearson_r_with_CER",
                         "Relationship", "Interpretation"])
        for param, r_val in sorted(correlations.items(),
                                   key=lambda x: abs(x[1]), reverse=True):
            if r_val > 0.3:
                rel = "DIRECT (higher → worse CER)"
            elif r_val < -0.3:
                rel = "INVERSE (higher → better CER)"
            else:
                rel = "WEAK / none"
            writer.writerow([param, r_val, rel, f"|r|={abs(r_val):.3f}"])
    print(f"  correlation_matrix.csv      → {corr_csv}")

    # ── Write human-readable sensitivity report ───────────────────────────────
    report_path = os.path.join(out_dir, "sensitivity_report.txt")
    top_n       = min(20, len(rows_sorted))
    lines = [
        "=" * 65,
        "  PART 1 — Parameter Sensitivity Report",
        f"  Generated : {_ts()}",
        f"  Grid size : {total} combinations  |  {failed_count} failed",
        f"  Runtime   : {total_elapsed:.1f}s",
        "=" * 65,
        "",
        "  ── Fixed parameters (not swept) ────────────────────────────",
    ]
    for k, v in fixed.items():
        lines.append(f"    {k:<24} = {v}")
    lines += [
        "",
        "  ── Swept grid ──────────────────────────────────────────────",
    ]
    for k, v in grid.items():
        lines.append(f"    {k:<24} : {v}")
    lines += [
        "",
        "  ── Top configurations by mean CER ──────────────────────────",
        f"  {'Rank':<5} {'CER%':<8} {'WER%':<8} {'sk':<4} "
        f"{'ck':<4} {'wp':<5} {'mst':<7}",
        "  " + "-" * 44,
    ]
    for rank, row in enumerate(rows_sorted[:top_n], 1):
        lines.append(
            f"  {rank:<5} {row['mean_cer']:<8.2f} {row['mean_wer']:<8.2f} "
            f"{row['smoothing_k']:<4} "
            f"{row['close_k']:<4} {row['window_pad']:<5} "
            f"{row['multi_seg_threshold']:<7}"
        )
    lines += [
        "",
        "  ── Best configuration ──────────────────────────────────────",
    ]
    if rows_sorted:
        best = rows_sorted[0]
        for k in param_keys:
            lines.append(f"    {k:<24} = {best[k]}")
        lines.append(f"    {'mean_cer':<24} = {best['mean_cer']:.2f}%")
        lines.append(f"    {'mean_wer':<24} = {best['mean_wer']:.2f}%")
    lines += [
        "",
        "  ── Parameter correlations with CER (Pearson r) ─────────────",
    ]
    for param, r_val in sorted(correlations.items(),
                                key=lambda x: abs(x[1]), reverse=True):
        bar       = "▮" * int(abs(r_val) * 20)
        direction = "+" if r_val > 0 else "-"
        lines.append(f"    {param:<24}  r={r_val:+.4f}  {direction}{bar}")
    lines += [
        "",
        "  ── Quadratic direction + sensitivity ───────────────────────",
        "  (direction: +1=increase to reduce CER, -1=decrease to reduce CER)",
    ]
    for param, info in param_directions.items():
        arrow = "→ increase ↑" if info["direction"] == 1 else "→ decrease ↓"
        lines.append(f"    {param:<24}  dir={info['direction']:+d}  "
                     f"sens={info['sensitivity']:.4f}  {arrow}")
    lines += [
        "",
        "  ── Inter-parameter relationship matrix ─────────────────────",
        "  (row A, col B: scaling factor — if A moves 1, B should move this much)",
    ]
    header_row = f"  {'':24}" + "".join(f"  {p[:8]:>10}" for p in param_keys)
    lines.append(header_row)
    for a in param_keys:
        vals = "".join(f"  {rel_matrix[a][b]:>10.3f}" for b in param_keys)
        lines.append(f"  {a:<24}{vals}")
    lines += [
        "",
        "  ── Output files ────────────────────────────────────────────",
        f"    per_image_cer.csv            — per-image CER for every combo",
        f"    sensitivity_results.csv      — full ranked table",
        f"    correlation_matrix.csv       — Pearson r correlations",
        f"    param_directions.json        — quadratic direction+sensitivity",
        f"    param_relationship_matrix.csv — inter-parameter scaling matrix",
        f"    sensitivity_report.txt       — this file",
        "=" * 65,
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  sensitivity_report.txt      → {report_path}")

    # ── best_params.json for Part 2 fallback ─────────────────────────────────
    if rows_sorted:
        best_params_path = os.path.join(out_dir, "best_params.json")
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(rows_sorted[0], f, ensure_ascii=False, indent=2)
        print(f"  best_params.json            → {best_params_path}")

    print(f"\n  [{_ts()}] Sensitivity analysis complete in {total_elapsed:.1f}s")
    if rows_sorted:
        print(f"  Best CER: {rows_sorted[0]['mean_cer']:.2f}% "
              f"(sk={rows_sorted[0]['smoothing_k']}, "
              f"ck={rows_sorted[0]['close_k']}, "
              f"wp={rows_sorted[0]['window_pad']}, "
              f"mst={rows_sorted[0]['multi_seg_threshold']})")
    print("=" * 65 + "\n")

    return rows_sorted


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Part 1: Parameter sensitivity grid search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input_folder",   default=INPUT_FOLDER)
    ap.add_argument("--label_csv",      default=LABEL_CSV)
    ap.add_argument("--model_path",     default=MODEL_PATH)
    ap.add_argument("--class_map",      default=CLASS_MAP)
    ap.add_argument("--work_root",      default=WORK_ROOT)
    ap.add_argument("--out_dir",        default=SENSITIVITY_OUT)
    ap.add_argument("--sample",         type=int, default=SENS_SAMPLE,
                    help="Images per grid run (keep small for speed)")
    ap.add_argument("--seed",           type=int, default=SENS_SEED)
    ap.add_argument("--quick",          action="store_true",
                    help="Use reduced QUICK_GRID for a fast sanity check")
    ap.add_argument("--no_word_spacer", action="store_true")
    args = ap.parse_args()

    active_grid = QUICK_GRID if args.quick else FULL_GRID
    run_sensitivity(
        grid           = active_grid,
        fixed          = FIXED_PARAMS,
        sample         = args.sample,
        seed           = args.seed,
        out_dir        = args.out_dir,
        input_folder   = args.input_folder,
        label_csv      = args.label_csv,
        model_path     = args.model_path,
        class_map      = args.class_map,
        base_work_root = args.work_root,
        word_spacer    = not args.no_word_spacer,
    )