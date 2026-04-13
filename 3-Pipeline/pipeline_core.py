# =============================================================================
# pipeline_core.py  —  Sinhala OCR  |  Pipeline Orchestrator
#
# This is the main entry point that calls all stages in sequence.
# It re-exports the public API so that part1/part2/part3 can continue
# importing from this single module with no changes.
#
# Architecture:
#   part1_sensitivity.py ─┐
#   part2_heuristic.py  ──┤──▸ pipeline_core.py (this file)
#   part3_inference.py  ──┘         │
#       ┌───────────────────────────┤
#       ▼                           ▼
#   stage1_config.py      stage2_preprocessing.py
#       ▼                           │
#   stage3_classification.py  ◂─────┘
#       ▼
#   stage4_tobedeclared.py  (placeholder — future refinement)
#       ▼
#   stage5_reporting.py
#
# Stage data flow:
# ─────────────────────────────────────────────────────────────────────
#
#   STAGE 2 — Preprocessing   (stage2_preprocessing.py)
#     IN:  all_files  : list[str]       — raw image paths
#           labels     : dict[str, str]  — {stem: ground_truth}
#           cfg        : PipelineConfig
#     OUT: list[dict]  — per-image metadata:
#           { stem, fname, ground_truth, orig_path, binary_path,
#             skel_path, segments_path, word_segments_path,
#             temp_dir, n_segments, n_words, params }
#
#   STAGE 3 — Classification  (stage3_classification.py)
#     IN:  stems        : list[str]         — from Stage 2 output
#           model        : EfficientNetV2-S  — loaded once
#           idx_to_class : dict[str, str]
#           cfg          : PipelineConfig
#     OUT: list[dict]  — per-image results:
#           { stem, ground_truth, predicted_text,
#             predicted_text_no_spaces, wer, cer,
#             n_segments, n_aksharas, n_multi_seg,
#             n_words, aksharas }
#
#   STAGE 4 — To Be Declared  (stage4_tobedeclared.py)
#     IN:  all_results : list[dict]  — from Stage 3 output
#           cfg         : PipelineConfig
#     OUT: list[dict]  — same format (pass-through for now)
#
#   STAGE 5 — Reporting       (stage5_reporting.py)
#     IN:  stems     : list[str]
#           work_root : str
#           elapsed_s : float
#           cfg       : PipelineConfig
#     OUT: list[dict]  — per-image summaries:
#           { stem, fname, predicted_text, ground_truth,
#             n_aksharas, n_words, wer, cer, html_path }
#
# Usage (standalone):
#   python pipeline_core.py
#   python pipeline_core.py --sample 20 --seed 7
#   python pipeline_core.py --fullset
# =============================================================================

from __future__ import annotations

import os
import sys
import json
import time
import random
import argparse

# ─── Stage imports ───────────────────────────────────────────────────────────
from stage1_config import (                             # noqa: F401
    # Path constants
    INPUT_FOLDER, LABEL_CSV, TESSERACT_CSV, MODEL_PATH, CLASS_MAP, WORK_ROOT,
    # Parameter defaults
    P_TARGET_HEIGHT, P_SMOOTHING_K, P_CLOSE_K, P_SKELETON_DIL,
    P_VALLEY_MIN_WIDTH, P_CHAR_CANVAS_SIZE, P_WINDOW_PAD,
    P_MULTI_SEG_THRESHOLD, P_WORD_SPACER_ENABLED, P_WORD_GAP_PX,
    TOP_K,
    # GPU config
    INFER_BATCH_SIZE, S1_WORKERS, PNG_POOL_WORKERS, CUDNN_BENCHMARK,
    # Normalisation tensors
    _NORM_MEAN, _NORM_STD,
    # Device
    _select_device, _DEVICE,
    # Config dataclass
    PipelineConfig,
)

from stage2_preprocessing import (                      # noqa: F401
    _load_labels, _save_png,
    _preprocess_sentence, _skeletonize_roi,
    _build_sentence_skeleton, _find_valley_segments, _find_word_groups,
    run_stage2_preprocessing,
)

from stage3_classification import (                     # noqa: F401
    _load_model, _warmup_model,
    _class_to_sinhala,
    _build_full_skeleton, _make_window_crop_np,
    _np_to_tensor_pinned, _predict_batch,
    _greedy_segment, _annotate_word_indices,
    _edit_distance, compute_cer, compute_wer,
    run_stage3_classification,
)

from stage4_tobedeclared import (                       # noqa: F401
    run_stage4_tobedeclared,
)

from stage5_reporting import (                          # noqa: F401
    _build_composite_png, _build_html,
    _write_segments_csv, _write_summary_csv,
    _write_master_summary, _write_batch_report,
    run_stage5_reporting, generate_flat_report,
)

# =============================================================================
# ██████████████████████  PUBLIC API  █████████████████████████████████████████
# =============================================================================

def run_pipeline(cfg:          PipelineConfig = None,
                 model         = None,
                 idx_to_class: dict = None,
                 skip_stage3:  bool = False,
                 silent:       bool = False):
    """
    Execute the full pipeline for a given PipelineConfig.

    Parameters
    ----------
    cfg          : PipelineConfig instance.
    model        : Pre-loaded EfficientNetV2-S.  If None, loaded from cfg.model_path.
    idx_to_class : {str_index: class_name}.  If None, loaded from cfg.class_map.
    skip_stage3  : If True, skip Stage 5 (reporting).
    silent       : If True, suppress banner prints.

    Returns
    -------
    _ResultList  — list of per-image dicts with .mean_cer and .mean_wer attributes.
    """
    if cfg is None:
        cfg = PipelineConfig()
    t0 = time.time()

    if not silent:
        print(f"\n{'=' * 62}")
        print(f"  Sinhala OCR Pipeline  |  EfficientNetV2-S  |  {_DEVICE}")
        print(f"{'=' * 62}")
        for k, v in cfg.as_param_dict().items():
            print(f"  {k:<24} = {v}")
        print(f"{'=' * 62}\n")

    # ── Load model + class map (once) ─────────────────────────────────────────
    if idx_to_class is None:
        with open(cfg.class_map, "r", encoding="utf-8") as f:
            idx_to_class = json.load(f)

    device = _DEVICE
    if model is None:
        model = _load_model(cfg.model_path, len(idx_to_class), device)

    # ── Collect images ────────────────────────────────────────────────────────
    valid_exts = {".png", ".jpg", ".jpeg"}
    all_files  = sorted([
        os.path.join(cfg.input_folder, fn)
        for fn in os.listdir(cfg.input_folder)
        if os.path.splitext(fn.lower())[1] in valid_exts
    ])
    if not cfg.fullset:
        rng       = random.Random(cfg.seed)
        all_files = rng.sample(all_files, min(cfg.sample, len(all_files)))
    if not all_files:
        print("  No images found — nothing to do.")
        return _ResultList([], 0.0, 0.0)

    labels = _load_labels(cfg.label_csv)

    # ── Work root / temp ──────────────────────────────────────────────────────
    work_root = cfg.work_root
    temp_root = os.path.join(work_root, "temp")
    os.makedirs(temp_root, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2 — Preprocessing
    #   IN:  raw image paths, labels dict, PipelineConfig
    #   OUT: list[dict] — per-image metadata (stem, paths to artefacts)
    # ──────────────────────────────────────────────────────────────────────────
    stage2_metas = run_stage2_preprocessing(
        cfg       = cfg,
        temp_root = temp_root,
        all_files = all_files,
        labels    = labels,
    )
    stems = [m["stem"] for m in stage2_metas]

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3 — Classification
    #   IN:  stems list, loaded model, device, idx_to_class, PipelineConfig
    #   OUT: list[dict] — per-image results (predicted_text, cer, wer, aksharas)
    # ──────────────────────────────────────────────────────────────────────────
    stage3_results = run_stage3_classification(
        cfg          = cfg,
        stems        = stems,
        work_root    = work_root,
        model        = model,
        device       = device,
        idx_to_class = idx_to_class,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 4 — To Be Declared  (placeholder — passes through unchanged)
    #   IN:  list[dict] — Stage 3 classification results
    #   OUT: list[dict] — same format (refined results in future)
    # ──────────────────────────────────────────────────────────────────────────
    stage4_results = run_stage4_tobedeclared(
        cfg         = cfg,
        all_results = stage3_results,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 5 — Reporting  (skipped if skip_stage3=True)
    #   IN:  stems list, work_root, elapsed_s, PipelineConfig
    #   OUT: list[dict] — per-image summaries (html_path, final cer/wer)
    # ──────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    if not skip_stage3:
        run_stage5_reporting(
            cfg       = cfg,
            stems     = stems,
            work_root = work_root,
            elapsed_s = elapsed,
        )

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    gt_results = [r for r in stage4_results if r.get("ground_truth")]
    mean_cer   = (sum(r.get("cer", 0) for r in gt_results) / len(gt_results)
                  if gt_results else 0.0)
    mean_wer   = (sum(r.get("wer", 0) for r in gt_results) / len(gt_results)
                  if gt_results else 0.0)

    if not silent:
        elapsed = time.time() - t0
        print(f"\n{'=' * 62}")
        print(f"  [OK]  Pipeline complete  -  {elapsed:.1f}s")
        if gt_results:
            print(f"  [OK]  Mean CER: {mean_cer:.2f}%   Mean WER: {mean_wer:.2f}%")
        print(f"{'=' * 62}\n")

    return _ResultList(stage4_results, mean_cer=mean_cer, mean_wer=mean_wer)


class _ResultList(list):
    """list subclass carrying aggregate metrics as attributes."""
    def __init__(self, iterable, mean_cer: float = 0.0, mean_wer: float = 0.0):
        super().__init__(iterable)
        self.mean_cer = mean_cer
        self.mean_wer = mean_wer

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Sinhala OCR Pipeline — EfficientNetV2-S",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input_folder",   default=INPUT_FOLDER)
    ap.add_argument("--label_csv",      default=LABEL_CSV)
    ap.add_argument("--model_path",     default=MODEL_PATH)
    ap.add_argument("--class_map",      default=CLASS_MAP)
    ap.add_argument("--work_root",      default=WORK_ROOT)
    ap.add_argument("--fullset",        action="store_true")
    ap.add_argument("--sample",         type=int, default=50)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--target_height",  type=int, default=P_TARGET_HEIGHT)
    ap.add_argument("--smoothing_k",    type=int, default=P_SMOOTHING_K)
    ap.add_argument("--close_k",        type=int, default=P_CLOSE_K)
    ap.add_argument("--skeleton_dil",   type=int, default=P_SKELETON_DIL)
    ap.add_argument("--valley_min_width", type=int, default=P_VALLEY_MIN_WIDTH)
    ap.add_argument("--window_pad",     type=int, default=P_WINDOW_PAD)
    ap.add_argument("--multi_seg_threshold", type=float, default=P_MULTI_SEG_THRESHOLD)
    ap.add_argument("--no_word_spacer", action="store_true")
    ap.add_argument("--word_gap_px",    type=int, default=P_WORD_GAP_PX)
    args = ap.parse_args()

    cfg = PipelineConfig(
        input_folder        = args.input_folder,
        label_csv           = args.label_csv,
        model_path          = args.model_path,
        class_map           = args.class_map,
        work_root           = args.work_root,
        fullset             = args.fullset,
        sample              = args.sample,
        seed                = args.seed,
        target_height       = args.target_height,
        smoothing_k         = args.smoothing_k,
        close_k             = args.close_k,
        skeleton_dil        = args.skeleton_dil,
        valley_min_width    = args.valley_min_width,
        window_pad          = args.window_pad,
        multi_seg_threshold = args.multi_seg_threshold,
        word_spacer_enabled = not args.no_word_spacer,
        word_gap_px         = args.word_gap_px,
    )
    run_pipeline(cfg)