# =============================================================================
# part3_inference.py  —  Phase 3: Dynamic Inference Pipeline
#
# Purpose
# -------
# Production-ready pipeline that uses the trained Multi-Output Random Forest
# (from Part 2) to predict all 4 preprocessing parameters simultaneously for
# each incoming image, then runs OCR recognition.
#
# Flow per image
# --------------
#  RAW image
#    │
#    ▼
#  Feature extraction  (14 image stats — same as Part 2)
#    │
#    ▼
#  Multi-output RF query  → predicts smoothing_k, close_k,
#                           window_pad, multi_seg_threshold
#    │
#    ▼
#  Merge predicted params with FIXED_PARAMS  → PipelineConfig
#    │
#    ▼
#  EfficientNetV2-S inference  (greedy multi-seg)
#    │
#    ▼
#  Output:  predicted Sinhala text + CER/WER + HTML/CSV report
#
# BUG FIXES vs original
# ----------------------
# 1. CRITICAL — CER always 0.0:
#    Original code set adapted_cfg.input_folder = os.path.dirname(img_path)
#    (the entire dataset folder), then used fullset=False, sample=1, seed=0.
#    random.sample(all_30k_files, 1, seed=0) picked a RANDOM image, not the
#    requested one.  The monkeypatched labels had no entry for that random
#    stem → ground_truth="" → compute_cer(pred, "") returns 0.0 by design.
#
# 2. FRAGILE monkeypatch:
#    Replacing pipeline_core._load_labels with a lambda is not thread-safe
#    and breaks re-entrant calls.  Replaced with the same isolated-folder +
#    private-label-CSV pattern used in Part 2's run_combo_on_image, which
#    is fully deterministic and requires no global mutation.
#
# 3. Wrong image processed:
#    The isolated-folder approach guarantees that run_pipeline processes
#    exactly one file — the one requested — by placing only that file (via
#    symlink or copy) in a private temp directory and using fullset=True.
#
# Usage
# -----
#   python part3_inference.py
#   python part3_inference.py --sample 50
#   python part3_inference.py --fullset
#   python part3_inference.py --image path/to/single.jpg
#   python part3_inference.py --no_tree     # use best fixed params from Part 1
#.\venv311\Scripts\python part3_inference.py --sample 50
# =============================================================================

from __future__ import annotations

import os
import sys
import json
import csv
import shutil
import time
import argparse
import warnings
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
    PipelineConfig, run_pipeline, compute_cer, compute_wer,
    _load_model, _load_labels,
    _DEVICE,
    INPUT_FOLDER, LABEL_CSV, TESSERACT_CSV, MODEL_PATH, CLASS_MAP, WORK_ROOT,
)

# Re-use feature extractor and key list from Part 2
from part2_heuristic import extract_image_features, FEATURE_KEYS

import torch

# =============================================================================
# TOP-OF-FILE CONFIGURATION  (adjust here)
# =============================================================================

PHASE2_OUT  = os.path.join(WORK_ROOT, "Info", "phase2_heuristic")
PHASE3_OUT  = os.path.join(WORK_ROOT, "Results")

# Paths to trained artefacts (resolved from PHASE2_OUT)
RF_MODEL_PATH = os.path.join(PHASE2_OUT, "random_forest_model.joblib")
SCALER_PATH   = os.path.join(PHASE2_OUT, "feature_scaler.joblib")
PROFILES_PATH = os.path.join(PHASE2_OUT, "param_profiles.json")

# Fallback: best fixed params from Part 1 (used when --no_tree)
FALLBACK_BEST_PARAMS_PATH = os.path.join(
    WORK_ROOT, "Info", "phase1_sensitivity", "best_params.json")

# Fixed parameters (never swept) — must match Part 1
FIXED_PARAMS = {
    "target_height":    384,
    "skeleton_dil":     1,
    "valley_min_width": 2,
    "word_gap_px":      35,
}

# Swept parameter names (must match Part 2)
SWEPT_PARAMS = ["smoothing_k", "close_k", "window_pad", "multi_seg_threshold"]

DEFAULT_SAMPLE = 500
DEFAULT_SEED   = 42

# =============================================================================
# HELPERS
# =============================================================================

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# ISOLATED SINGLE-IMAGE PIPELINE RUNNER
# =============================================================================

def _run_pipeline_single_image(
    img_path:     str,
    ground_truth: str,
    adapted_cfg:  PipelineConfig,
    model,
    idx_to_class: dict,
    run_idx:      int,
) -> dict:
    """
    Run the OCR pipeline on EXACTLY one image with a fully isolated environment.

    Strategy (identical to Part 2's run_combo_on_image — proven correct):
      1. Create a private temp directory containing ONLY this image
         (symlinked or copied).
      2. Write a private label CSV containing ONLY this stem's ground truth.
      3. Set fullset=True on the config so the pipeline processes all files
         in the private folder — which is exactly one file.
      4. No monkeypatching of any global state.

    This guarantees:
      - The pipeline always processes the requested image, never a random one.
      - Ground truth is read from a real CSV file, not a patched function.
      - CER/WER are computed correctly against the real ground truth.

    Returns the matched per-image result dict (with stem, cer, wer,
    predicted_text, ground_truth keys), or an empty dict on failure.
    """
    fname = os.path.basename(img_path)
    stem  = os.path.splitext(fname)[0]

    # Private working tree for this probe
    run_work      = adapted_cfg.work_root   # already set per-image by caller
    img_tmp_dir   = os.path.join(run_work, "img_input")
    label_tmp_path = os.path.join(run_work, "labels.csv")
    os.makedirs(img_tmp_dir, exist_ok=True)

    # ── Place ONLY this image in the private input folder ─────────────────────
    dst_img = os.path.join(img_tmp_dir, fname)
    if not os.path.exists(dst_img):
        try:
            os.symlink(os.path.abspath(img_path), dst_img)
        except (OSError, NotImplementedError):
            shutil.copy2(img_path, dst_img)

    # ── Write a private label CSV with ONLY this image's ground truth ─────────
    with open(label_tmp_path, "w", encoding="utf-8-sig", newline="") as f:
        gt_escaped = ground_truth.replace('"', '""')
        f.write(f'{stem},"{gt_escaped}"\n')

    # ── Build the isolated PipelineConfig ─────────────────────────────────────
    isolated_cfg = PipelineConfig(
        input_folder        = img_tmp_dir,       # ← ONLY this image's folder
        label_csv           = label_tmp_path,    # ← ONLY this image's GT
        model_path          = adapted_cfg.model_path,
        class_map           = adapted_cfg.class_map,
        work_root           = run_work,
        word_spacer_enabled = adapted_cfg.word_spacer_enabled,
        fullset             = True,              # ← process ALL files in img_tmp_dir (= 1)
        sample              = 1,                 # safety cap (fullset overrides)
        seed                = 0,
        target_height       = adapted_cfg.target_height,
        smoothing_k         = adapted_cfg.smoothing_k,
        close_k             = adapted_cfg.close_k,
        skeleton_dil        = adapted_cfg.skeleton_dil,
        valley_min_width    = adapted_cfg.valley_min_width,
        window_pad          = adapted_cfg.window_pad,
        multi_seg_threshold = adapted_cfg.multi_seg_threshold,
        word_gap_px         = adapted_cfg.word_gap_px,
    )

    results = run_pipeline(
        cfg          = isolated_cfg,
        model        = model,
        idx_to_class = idx_to_class,
        skip_stage3  = True,
        silent       = True,
    )

    # ── Find the result for this specific stem ────────────────────────────────
    matched = next((r for r in results if r.get("stem") == stem), None)
    if matched is None:
        matched = results[0] if results else {}

    return matched


# =============================================================================
# META-OPTIMIZER  (multi-output RF)
# =============================================================================

class MetaOptimizer:
    """
    Wraps the trained multi-output RF.
    predict_params(img_path) → dict of all 4 swept parameter values.
    """

    def __init__(self,
                 rf_path:     str = RF_MODEL_PATH,
                 scaler_path: str = SCALER_PATH):
        try:
            import joblib
        except ImportError:
            print("  ERROR: joblib not installed.  Run: pip install joblib scikit-learn")
            sys.exit(1)

        for path, label in [(rf_path, "random_forest_model.joblib"),
                             (scaler_path, "feature_scaler.joblib")]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{label} not found: {path}\n"
                    "Run part2_heuristic.py first.")

        self.clf    = joblib.load(rf_path)
        self.scaler = joblib.load(scaler_path)
        print(f"  [MetaOptimizer] Loaded multi-output RF from: {rf_path}")

    def predict_params(self, img_path: str) -> tuple[dict, dict]:
        """
        Extract features and predict all 4 swept parameter values.
        Returns (predicted_params_dict, features_dict).
        Falls back to grid medians on feature extraction failure.
        """
        from part2_heuristic import SWEPT_GRID

        features = extract_image_features(img_path)
        if features is None:
            fallback = {p: SWEPT_GRID[p][len(SWEPT_GRID[p]) // 2]
                        for p in SWEPT_PARAMS}
            return fallback, {}

        x        = np.array([[features[k] for k in FEATURE_KEYS]], dtype=float)
        x_scaled = self.scaler.transform(x)

        # MultiOutputClassifier.predict returns shape (n_samples, n_outputs)
        y_pred = self.clf.predict(x_scaled)[0]   # shape (4,)

        predicted = {}
        for i, p in enumerate(SWEPT_PARAMS):
            raw_val = float(y_pred[i])
            # Snap to nearest valid grid value (safety guard)
            grid_vals    = SWEPT_GRID[p]
            predicted[p] = min(grid_vals, key=lambda g: abs(g - raw_val))

        return predicted, features

    def build_adapted_config(self,
                              img_path:  str,
                              base_cfg:  PipelineConfig,
                              run_work:  str) -> tuple[PipelineConfig, dict]:
        """
        Return (adapted_config, features_dict).
        Merges predicted swept params with FIXED_PARAMS.
        The returned config has work_root set to run_work but does NOT
        set input_folder or label_csv — those are set by the caller's
        isolated runner.
        """
        predicted, features = self.predict_params(img_path)
        merged = {**FIXED_PARAMS, **predicted}

        adapted = PipelineConfig(
            input_folder        = base_cfg.input_folder,   # placeholder; overridden
            label_csv           = base_cfg.label_csv,      # placeholder; overridden
            model_path          = base_cfg.model_path,
            class_map           = base_cfg.class_map,
            work_root           = run_work,
            word_spacer_enabled = base_cfg.word_spacer_enabled,
            fullset             = True,
            sample              = 1,
            seed                = 0,
            target_height       = merged.get("target_height",       384),
            smoothing_k         = int(merged.get("smoothing_k",     3)),
            close_k             = int(merged.get("close_k",         3)),
            skeleton_dil        = int(merged.get("skeleton_dil",    1)),
            valley_min_width    = int(merged.get("valley_min_width", 2)),
            window_pad          = int(merged.get("window_pad",      12)),
            multi_seg_threshold = float(merged.get("multi_seg_threshold", 97.0)),
            word_gap_px         = int(merged.get("word_gap_px",     35)),
        )
        return adapted, features


# =============================================================================
# FALLBACK OPTIMIZER  (fixed best params from Part 1)
# =============================================================================

class FallbackOptimizer:
    """Used when --no_tree is set or trained artefacts are missing."""

    def __init__(self, best_params_path: str = FALLBACK_BEST_PARAMS_PATH):
        self._params: dict = {}
        if os.path.exists(best_params_path):
            with open(best_params_path, "r", encoding="utf-8") as f:
                self._params = json.load(f)
            print(f"  [FallbackOptimizer] Using best params from: {best_params_path}")
            print(f"    CER: {self._params.get('mean_cer', '?')}%")
        else:
            print(f"  [FallbackOptimizer] No best_params.json found — "
                  f"using pipeline defaults.")

    def build_adapted_config(self,
                              img_path: str,
                              base_cfg: PipelineConfig,
                              run_work: str) -> tuple[PipelineConfig, dict]:
        features = extract_image_features(img_path) or {}
        merged   = {**FIXED_PARAMS}
        for p in SWEPT_PARAMS:
            if p in self._params:
                merged[p] = self._params[p]

        adapted = PipelineConfig(
            input_folder        = base_cfg.input_folder,   # placeholder; overridden
            label_csv           = base_cfg.label_csv,      # placeholder; overridden
            model_path          = base_cfg.model_path,
            class_map           = base_cfg.class_map,
            work_root           = run_work,
            word_spacer_enabled = base_cfg.word_spacer_enabled,
            fullset             = True,
            sample              = 1,
            seed                = 0,
            target_height       = int(merged.get("target_height",       384)),
            smoothing_k         = int(merged.get("smoothing_k",         3)),
            close_k             = int(merged.get("close_k",             3)),
            skeleton_dil        = int(merged.get("skeleton_dil",        1)),
            valley_min_width    = int(merged.get("valley_min_width",    2)),
            window_pad          = int(merged.get("window_pad",          12)),
            multi_seg_threshold = float(merged.get("multi_seg_threshold", 97.0)),
            word_gap_px         = int(merged.get("word_gap_px",         35)),
        )
        return adapted, features


# =============================================================================
# PER-IMAGE DYNAMIC INFERENCE  (orchestrator)
# =============================================================================

def run_single_image_dynamic(img_path:     str,
                               ground_truth: str,
                               optimizer,
                               model,
                               idx_to_class: dict,
                               base_cfg:    PipelineConfig,
                               run_idx:     int,
                               tess_text:   Optional[str] = None) -> dict:
    """
    Orchestrate the full dynamic pipeline for ONE image.

    1. Ask the optimizer for adapted parameters.
    2. Delegate to _run_pipeline_single_image, which uses an isolated
       folder + private label CSV — no monkeypatching, no global mutation.
    3. Return a fully populated result dict.
    """
    fname    = os.path.basename(img_path)
    stem     = os.path.splitext(fname)[0]
    run_work = os.path.join(base_cfg.work_root, "Results", stem)

    # Build the parameter-adapted config (input_folder/label_csv are
    # placeholders here; _run_pipeline_single_image overwrites them)
    adapted_cfg, features = optimizer.build_adapted_config(
        img_path = img_path,
        base_cfg = base_cfg,
        run_work = run_work,
    )

    print(f"    [{run_idx}] {fname}  →  "
          f"sk={adapted_cfg.smoothing_k} ck={adapted_cfg.close_k} "
          f"wp={adapted_cfg.window_pad} mst={adapted_cfg.multi_seg_threshold}")

    try:
        matched = _run_pipeline_single_image(
            img_path     = img_path,
            ground_truth = ground_truth,
            adapted_cfg  = adapted_cfg,
            model        = model,
            idx_to_class = idx_to_class,
            run_idx      = run_idx,
        )

        cer            = matched.get("cer",            0.0)
        wer            = matched.get("wer",            0.0)
        predicted_text = matched.get("predicted_text", "")

        # ── Tesseract Comparison ──────────────────────────────────────────────
        tess_cer, tess_wer = None, None
        if ground_truth and tess_text is not None:
            tess_cer = compute_cer(tess_text, ground_truth)
            tess_wer = compute_wer(tess_text, ground_truth)

        # ── Stage 5: Reporting (Flatter structure) ────────────────────────────
        from pipeline_core import generate_flat_report
        tess_data = {"text": tess_text, "cer": tess_cer, "wer": tess_wer}
        generate_flat_report(stem, run_work, run_work, adapted_cfg, tess_data=tess_data)

        # ── Cleanup intermediate artefacts ────────────────────────────────────
        # Keep only the flat files in run_work
        for sub in ["temp", "img_input"]:
            p = os.path.join(run_work, sub)
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)
        lp = os.path.join(run_work, "labels.csv")
        if os.path.exists(lp):
            os.remove(lp)

        print(f"           predicted: {predicted_text or '—'}  "
              f"CER: {cer:.1f}%  WER: {wer:.1f}%")

    except Exception as exc:
        import traceback
        print(f"           ERROR: {exc}")
        traceback.print_exc()
        cer, wer, predicted_text = 100.0, 100.0, ""
        matched = {}
        tess_cer, tess_wer = None, None

    return {
        "stem":           stem,
        "fname":          fname,
        "img_path":       img_path,
        "ground_truth":   ground_truth,
        "predicted_text": predicted_text,
        "cer":            cer,
        "wer":            wer,
        "params_used":    adapted_cfg.as_param_dict(),
        "tess_text":      tess_text,
        "tess_cer":       tess_cer,
        "tess_wer":       tess_wer,
        "features":       features,
    }


# =============================================================================
# BATCH REPORTS
# =============================================================================

def _load_tesseract_results(path: str) -> dict[str, str]:
    """Load Tesseract results from CSV (file_name,text)."""
    results = {}
    if not os.path.exists(path):
        return results
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stem = row.get("file_name", "").split(".")[0]
                results[stem] = row.get("text", "")
    except Exception as e:
        print(f"  [ERROR] Could not load Tesseract CSV: {e}")
    return results

def write_phase3_report(all_results:    list[dict],
                          out_dir:       str,
                          elapsed_s:     float,
                          optimizer_name: str) -> None:
    gt_results = [r for r in all_results if r.get("ground_truth")]
    mean_cer   = (sum(r["cer"] for r in gt_results) / len(gt_results)
                  if gt_results else None)
    mean_wer   = (sum(r["wer"] for r in gt_results) / len(gt_results)
                  if gt_results else None)

    lines = [
        "=" * 65,
        "  PART 3 — Dynamic Inference Report  (EfficientNetV2-S)",
        f"  Generated    : {_ts()}",
        f"  Optimizer    : {optimizer_name}",
        f"  Total images : {len(all_results)}",
        f"  With GT      : {len(gt_results)}",
        f"  Elapsed      : {elapsed_s:.1f}s  "
        f"({elapsed_s/max(len(all_results),1):.1f}s/image)",
        "=" * 65,
        "",
        "  ── Aggregate accuracy ──────────────────────────────────────",
        f"  Mean CER : {mean_cer:.2f}%" if mean_cer is not None else "  Mean CER : N/A",
        f"  Mean WER : {mean_wer:.2f}%" if mean_wer is not None else "  Mean WER : N/A",
        "",
        "  ── Per-image summary ───────────────────────────────────────",
    ]
    for r in all_results:
        cer_str = f"{r['cer']:.2f}%" if r["ground_truth"] else "N/A"
        wer_str = f"{r['wer']:.2f}%" if r["ground_truth"] else "N/A"
        p       = r.get("params_used", {})
        lines.append(
            f"  {r['fname']:<40} CER {cer_str:>8}  WER {wer_str:>8}  "
            f"sk={p.get('smoothing_k','?')} "
            f"ck={p.get('close_k','?')} "
            f"wp={p.get('window_pad','?')} "
            f"mst={p.get('multi_seg_threshold','?')}"
        )
    lines += ["", "=" * 65]

    report_path = os.path.join(out_dir, "run_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  run_summary.txt    → {report_path}")


def write_phase3_csv(all_results: list[dict], out_dir: str) -> None:
    csv_path   = os.path.join(out_dir, "run_results_summary.csv")
    if not all_results:
        return
    param_keys = list(all_results[0]["params_used"].keys())
    fieldnames = (["fname", "stem", "ground_truth", "predicted_text",
                   "cer", "wer", "tess_text", "tess_cer", "tess_wer"] + param_keys)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, "") for k in ["fname", "stem", "ground_truth",
                                               "predicted_text", "cer", "wer"]}
            row.update(r.get("params_used", {}))
            writer.writerow(row)
    print(f"  run_results.csv    → {csv_path}")


def write_phase3_html_summary(all_results: list[dict], out_dir: str, elapsed: float) -> None:
    html_path = os.path.join(out_dir, "run_dashboard.html")
    gt_res    = [r for r in all_results if r.get("ground_truth")]
    m_cer     = sum(r["cer"] for r in gt_res) / len(gt_res) if gt_res else 0.0
    m_wer     = sum(r["wer"] for r in gt_res) / len(gt_res) if gt_res else 0.0
    t_cer     = sum(r.get("tess_cer", 100.0) for r in gt_res if r.get("tess_cer") is not None) / len([r for r in gt_res if r.get("tess_cer") is not None]) if any(r.get("tess_cer") is not None for r in gt_res) else 0.0

    rows = ""
    for r in all_results:
        cer_col = "#4ecca3" if r['cer'] < 15 else ("#f0a500" if r['cer'] < 40 else "#e94560")
        rows += f"""
        <tr>
            <td><a href="{r['stem']}/index.html" style="color:#4ecca3">{r['fname']}</a></td>
            <td class="si">{r['predicted_text']}</td>
            <td class="si" style="color:#888">{r['ground_truth']}</td>
            <td style="color:{cer_col}; font-weight:bold">{r['cer']:.1f}%</td>
            <td style="color:#aaa">{(r.get('tess_cer') or 0.0):.1f}%</td>
            <td>{r['wer']:.1f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>Inference Dashboard</title>
    <style>
        body {{ font-family: sans-serif; background: #0f0f1b; color: #e0e0e0; padding: 40px; }}
        h1 {{ color: #e94560; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ background: #16213e; padding: 20px; border-radius: 10px; flex: 1; text-align: center; border: 1px solid #1a2a40; }}
        .stat div {{ font-size: 32px; font-weight: bold; color: #4ecca3; }}
        .stat span {{ font-size: 12px; color: #888; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; background: #16213e; border-radius: 10px; overflow: hidden; }}
        th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #1a2a40; }}
        th {{ background: #1a2a40; color: #888; font-size: 12px; text-transform: uppercase; }}
        .si {{ font-size: 18px; }}
        a {{ text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style></head><body>
    <h1>OCR Inference Dashboard</h1>
    <div class="stats">
        <div class="stat"><div>{len(all_results)}</div><span>Total Images</span></div>
        <div class="stat"><div>{m_cer:.1f}%</div><span>Mean CER (Ours)</span></div>
        <div class="stat"><div style="color:#aaa">{t_cer:.1f}%</div><span>Mean CER (Tess)</span></div>
        <div class="stat"><div>{elapsed:.1f}s</div><span>Total Time</span></div>
    </div>
    <table>
        <tr><th>Image</th><th>Prediction</th><th>Ground Truth</th><th>Our CER</th><th>Tess CER</th><th>WER</th></tr>
        {rows}
    </table>
    </body></html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  run_dashboard.html → {html_path}")


# =============================================================================
# MAIN ENTRY
# =============================================================================

def run_inference(input_folder:  str  = INPUT_FOLDER,
                   label_csv:     str  = LABEL_CSV,
                   model_path:    str  = MODEL_PATH,
                   class_map:     str  = CLASS_MAP,
                   work_root:     str  = WORK_ROOT,
                   out_dir:       str  = PHASE3_OUT,
                   fullset:       bool = False,
                   sample:        int  = DEFAULT_SAMPLE,
                   seed:          int  = DEFAULT_SEED,
                   word_spacer:   bool = True,
                   use_tree:      bool = True,
                   single_image:  Optional[str] = None) -> list[dict]:
    """
    Run the full dynamic inference pipeline.
    Returns list of per-image result dicts.
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    print(f"\n{'='*65}")
    print(f"  Part 3 — Dynamic Inference Pipeline  (EfficientNetV2-S)")
    print(f"  Started : {_ts()}")
    print(f"  Use meta-tree : {'YES (multi-output RF)' if use_tree else 'NO (fixed best params)'}")
    print(f"{'='*65}\n")

    # ── Load meta-optimizer ───────────────────────────────────────────────────
    if use_tree:
        try:
            optimizer      = MetaOptimizer()
            optimizer_name = "MetaOptimizer (multi-output RF)"
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            print("  Falling back to FallbackOptimizer (best fixed params).")
            optimizer      = FallbackOptimizer()
            optimizer_name = "FallbackOptimizer"
    else:
        optimizer      = FallbackOptimizer()
        optimizer_name = "FallbackOptimizer (--no_tree)"

    # ── Load model once ───────────────────────────────────────────────────────
    with open(class_map, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    device = _DEVICE
    model  = _load_model(model_path, len(idx_to_class), device)

    # ── Collect images ────────────────────────────────────────────────────────
    labels    = _load_labels(label_csv)
    tess_dict = _load_tesseract_results(TESSERACT_CSV)

    if single_image:
        img_paths = [single_image]
    else:
        valid_exts = {".png", ".jpg", ".jpeg"}
        import random as _random
        all_files = sorted([
            os.path.join(input_folder, fn)
            for fn in os.listdir(input_folder)
            if os.path.splitext(fn.lower())[1] in valid_exts
        ])
        if not fullset:
            _random.seed(seed)
            img_paths = _random.sample(all_files, min(sample, len(all_files)))
        else:
            img_paths = all_files

    print(f"  [{_ts()}] Images to process: {len(img_paths)}\n")

    # ── Base config — carries shared paths/settings; per-image runner
    #    creates its own isolated config from this template ────────────────────
    base_cfg = PipelineConfig(
        input_folder        = input_folder,
        label_csv           = label_csv,
        model_path          = model_path,
        class_map           = class_map,
        work_root           = work_root,
        word_spacer_enabled = word_spacer,
    )

    # ── Process each image ────────────────────────────────────────────────────
    all_results: list[dict] = []
    for run_idx, img_path in enumerate(img_paths, 1):
        fname        = os.path.basename(img_path)
        stem         = os.path.splitext(fname)[0]
        ground_truth = labels.get(stem, "")
        tess_text    = tess_dict.get(stem)

        result = run_single_image_dynamic(
            img_path     = img_path,
            ground_truth = ground_truth,
            optimizer    = optimizer,
            model        = model,
            idx_to_class = idx_to_class,
            base_cfg     = base_cfg,
            run_idx      = run_idx,
            tess_text    = tess_text,
        )
        all_results.append(result)

    # ── Reports ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n  [{_ts()}] Writing reports …")
    write_phase3_report(all_results, out_dir, elapsed, optimizer_name)
    if all_results:
        write_phase3_csv(all_results, out_dir)
        write_phase3_html_summary(all_results, out_dir, elapsed)

    # ── Summary ───────────────────────────────────────────────────────────────
    gt_res   = [r for r in all_results if r.get("ground_truth")]
    mean_cer = sum(r["cer"] for r in gt_res) / len(gt_res) if gt_res else None
    mean_wer = sum(r["wer"] for r in gt_res) / len(gt_res) if gt_res else None

    print(f"\n{'='*65}")
    print(f"  ✓  Dynamic inference complete at {_ts()}")
    print(f"  ✓  Elapsed : {elapsed:.1f}s")
    if mean_cer is not None:
        print(f"  ✓  Mean CER : {mean_cer:.2f}%   Mean WER : {mean_wer:.2f}%")
    print(f"  ✓  Results  : {out_dir}")
    print(f"{'='*65}\n")

    return all_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Part 3: Dynamic inference with multi-output meta-optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input_folder",   default=INPUT_FOLDER)
    ap.add_argument("--label_csv",      default=LABEL_CSV)
    ap.add_argument("--model_path",     default=MODEL_PATH)
    ap.add_argument("--class_map",      default=CLASS_MAP)
    ap.add_argument("--work_root",      default=WORK_ROOT)
    ap.add_argument("--out_dir",        default=PHASE3_OUT)
    ap.add_argument("--fullset",        action="store_true")
    ap.add_argument("--sample",         type=int, default=DEFAULT_SAMPLE)
    ap.add_argument("--seed",           type=int, default=DEFAULT_SEED)
    ap.add_argument("--no_word_spacer", action="store_true")
    ap.add_argument("--no_tree",        action="store_true",
                    help="Skip meta-tree; use best fixed params from Part 1")
    ap.add_argument("--image",          default=None,
                    help="Path to a single image (overrides sample/fullset)")
    args = ap.parse_args()

    run_inference(
        input_folder = args.input_folder,
        label_csv    = args.label_csv,
        model_path   = args.model_path,
        class_map    = args.class_map,
        work_root    = args.work_root,
        out_dir      = args.out_dir,
        fullset      = args.fullset,
        sample       = args.sample,
        seed         = args.seed,
        word_spacer  = not args.no_word_spacer,
        use_tree     = not args.no_tree,
        single_image = args.image,
    )