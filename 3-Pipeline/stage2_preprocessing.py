# =============================================================================
# stage2_preprocessing.py  —  Image Preprocessing & Skeletonisation
#
# Handles the entire Stage-2 preprocessing pipeline:
#   raw image → greyscale → binary → skeleton → valley segments → word groups
#
# INPUT:
#   - all_files   : list[str]       — absolute paths to input images
#   - labels      : dict[str, str]  — {stem: ground_truth_text}
#   - cfg         : PipelineConfig  — tunable parameters
#   - temp_root   : str             — directory for intermediate artefacts
#
# OUTPUT (from run_stage2_preprocessing):
#   - list[dict]  — per-image metadata dicts, each containing:
#       stem              : str   — filename without extension
#       fname             : str   — original filename with extension
#       ground_truth      : str   — label text (may be empty)
#       orig_path         : str   — path to copied original image
#       binary_path       : str   — path to saved binary image
#       skel_path         : str   — path to saved skeleton image
#       segments_path     : str   — path to segments.json
#       word_segments_path: str|None — path to word_segments.json (if enabled)
#       temp_dir          : str   — per-image temp directory
#       n_segments        : int   — number of valley segments found
#       n_words           : int   — number of word groups found
#       params            : dict  — preprocessing parameters used
# =============================================================================

from __future__ import annotations

import os
import json
import shutil
import concurrent.futures

import cv2
import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize

from stage1_config import PipelineConfig, S1_WORKERS

# =============================================================================
# HELPERS
# =============================================================================

def _load_labels(path: str) -> dict:
    """Load label CSV: each line is  stem,ground_truth_text."""
    labels = {}
    if not os.path.exists(path):
        print(f"  [WARN] Label file not found: {path}")
        return labels
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx = line.index(",")
            labels[line[:idx].strip()] = line[idx + 1:].strip().strip('"')
    return labels


def _save_png(numpy_img: np.ndarray, path: str) -> None:
    """Encode and write a numpy image as PNG (Unicode-safe)."""
    _, buf = cv2.imencode(".png", numpy_img)
    with open(path, "wb") as fh:
        fh.write(buf)

# =============================================================================
# SKELETON PIPELINE
# =============================================================================

def _preprocess_sentence(src_img: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Convert raw image → resized binary (black text on white)."""
    gray = (cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
            if len(src_img.shape) == 3 else src_img.copy())
    h, w     = gray.shape
    scale    = cfg.target_height / h
    new_size = (max(1, int(w * scale)), cfg.target_height)
    resized  = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)
    k        = cfg.smoothing_k if cfg.smoothing_k % 2 == 1 else cfg.smoothing_k + 1
    blurred  = cv2.GaussianBlur(resized, (k, k), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    return binary


def _skeletonize_roi(roi: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Skeletonize a single ROI using morphological close + skimage."""
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (cfg.close_k, cfg.close_k))
    healed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k)
    skel   = sk_skeletonize(healed > 0).astype(np.uint8) * 255
    if cfg.skeleton_dil > 0:
        skel = cv2.dilate(skel, np.ones((3, 3), np.uint8),
                          iterations=cfg.skeleton_dil)
    return skel


def _build_sentence_skeleton(binary: np.ndarray,
                              cfg: PipelineConfig) -> np.ndarray:
    """Build a full-sentence skeleton by skeletonizing each contour ROI."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(binary)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch < 30:
            continue
        canvas[y:y + ch, x:x + cw] = cv2.bitwise_or(
            canvas[y:y + ch, x:x + cw],
            _skeletonize_roi(binary[y:y + ch, x:x + cw], cfg))
    return canvas


def _find_valley_segments(skeleton_canvas: np.ndarray,
                           cfg: PipelineConfig) -> list:
    """Find character-level segments via vertical projection valleys."""
    projection = (skeleton_canvas > 0).sum(axis=0).astype(int)
    W          = projection.shape[0]
    in_seg     = False
    seg_start  = 0
    segments   = []
    for x in range(W):
        if projection[x] > 0 and not in_seg:
            seg_start = x
            in_seg    = True
        elif projection[x] == 0 and in_seg:
            gap_end = x
            while gap_end < W and projection[gap_end] == 0:
                gap_end += 1
            if gap_end - x >= cfg.valley_min_width:
                segments.append([seg_start, x])
                in_seg = False
    if in_seg:
        segments.append([seg_start, W])
    return segments


def _find_word_groups(segments: list,
                      skeleton_canvas: np.ndarray,
                      cfg: PipelineConfig) -> list:
    """Group segments into words based on gap width threshold."""
    if not segments:
        return []
    projection   = (skeleton_canvas > 0).sum(axis=0).astype(int)
    word_groups  = []
    current_word = [0]
    for seg_i in range(1, len(segments)):
        gap_start = segments[seg_i - 1][1]
        gap_end   = segments[seg_i][0]
        gap_width = int((projection[gap_start:gap_end] == 0).sum())
        if gap_width >= cfg.word_gap_px:
            word_groups.append({
                "word_index"  : len(word_groups),
                "seg_indices" : current_word[:],
                "x_start"     : segments[current_word[0]][0],
                "x_end"       : segments[current_word[-1]][1],
            })
            current_word = [seg_i]
        else:
            current_word.append(seg_i)
    word_groups.append({
        "word_index"  : len(word_groups),
        "seg_indices" : current_word[:],
        "x_start"     : segments[current_word[0]][0],
        "x_end"       : segments[current_word[-1]][1],
    })
    return word_groups

# =============================================================================
# PROCESS ONE IMAGE
# =============================================================================

def _process_one(img_path: str,
                 ground_truth: str,
                 temp_root: str,
                 cfg: PipelineConfig) -> dict:
    """Process a single image through the preprocessing pipeline (CPU-only, thread-safe)."""
    fname = os.path.basename(img_path)
    stem  = os.path.splitext(fname)[0]
    out   = os.path.join(temp_root, stem)
    os.makedirs(out, exist_ok=True)

    raw = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {img_path}")

    orig_out = os.path.join(out, fname)
    shutil.copy2(img_path, orig_out)

    binary          = _preprocess_sentence(img, cfg)
    skeleton_canvas = _build_sentence_skeleton(binary, cfg)

    binary_path = os.path.join(out, "binary.png")
    skel_path   = os.path.join(out, "skeleton.png")
    _save_png(binary,                           binary_path)
    _save_png(cv2.bitwise_not(skeleton_canvas), skel_path)

    segments      = _find_valley_segments(skeleton_canvas, cfg)
    segments_path = os.path.join(out, "segments.json")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f)

    word_segments_path = None
    n_words            = 0
    if cfg.word_spacer_enabled:
        word_groups        = _find_word_groups(segments, skeleton_canvas, cfg)
        word_segments_path = os.path.join(out, "word_segments.json")
        with open(word_segments_path, "w", encoding="utf-8") as f:
            json.dump(word_groups, f, ensure_ascii=False, indent=2)
        n_words = len(word_groups)

    meta = {
        "stem":                stem,
        "fname":               fname,
        "ground_truth":        ground_truth,
        "orig_path":           orig_out,
        "binary_path":         binary_path,
        "skel_path":           skel_path,
        "segments_path":       segments_path,
        "word_segments_path":  word_segments_path,
        "temp_dir":            out,
        "n_segments":          len(segments),
        "n_words":             n_words,
        "word_spacer_enabled": cfg.word_spacer_enabled,
        "word_gap_px":         cfg.word_gap_px,
        "params": {
            "TARGET_HEIGHT":    cfg.target_height,
            "SMOOTHING_K":      cfg.smoothing_k,
            "CLOSE_K":          cfg.close_k,
            "SKELETON_DIL":     cfg.skeleton_dil,
            "VALLEY_MIN_WIDTH": cfg.valley_min_width,
        },
    }
    meta_path = os.path.join(out, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta

# =============================================================================
# RUN STAGE 2 — PREPROCESSING
# =============================================================================

def run_stage2_preprocessing(cfg: PipelineConfig,
                              temp_root: str,
                              all_files: list,
                              labels: dict) -> list:
    """
    Run preprocessing on all files using a thread pool.

    Entirely CPU-bound (OpenCV + skimage).  Parallelised using S1_WORKERS
    threads.  Threads are safe because OpenCV and skimage release the GIL.
    """
    print(f"  [Stage 2 – Preprocessing] Processing {len(all_files)} image(s) "
          f"with {S1_WORKERS} worker thread(s) …")
    if cfg.word_spacer_enabled:
        print(f"  [Stage 2 – Preprocessing] Word spacer : ENABLED  (gap ≥ {cfg.word_gap_px} px)")
    else:
        print(f"  [Stage 2 – Preprocessing] Word spacer : DISABLED")

    def _worker(img_path: str) -> dict:
        fname        = os.path.basename(img_path)
        stem         = os.path.splitext(fname)[0]
        ground_truth = labels.get(stem, "")
        return _process_one(img_path, ground_truth, temp_root, cfg)

    metas        = []
    n            = len(all_files)
    completed    = 0
    failed_stems = []

    workers = min(S1_WORKERS, n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_path = {pool.submit(_worker, p): p for p in all_files}
        for future in concurrent.futures.as_completed(future_to_path):
            img_path = future_to_path[future]
            fname    = os.path.basename(img_path)
            completed += 1
            try:
                meta = future.result()
                info = (f"  words: {meta['n_words']}"
                        if cfg.word_spacer_enabled else "")
                print(f"    [{completed}/{n}] {fname}  "
                      f"segs: {meta['n_segments']}{info}")
                metas.append(meta)
            except Exception as exc:
                import traceback
                print(f"    [{completed}/{n}] {fname}  ERROR: {exc}")
                traceback.print_exc()
                failed_stems.append(fname)

    if failed_stems:
        print(f"  [Stage 2 – Preprocessing] {len(failed_stems)} image(s) failed: {failed_stems}")

    # Preserve deterministic ordering (as_completed is unordered)
    path_order = {p: i for i, p in enumerate(all_files)}
    metas.sort(key=lambda m: path_order.get(
        next((p for p in all_files
              if os.path.splitext(os.path.basename(p))[0] == m["stem"]),
             ""), 0))

    image_list_path = os.path.join(temp_root, "image_list.json")
    with open(image_list_path, "w", encoding="utf-8") as f:
        json.dump([m["stem"] for m in metas], f, ensure_ascii=False, indent=2)

    print(f"  [Stage 2 – Preprocessing] Done — temp artefacts in: {temp_root}\n")
    return metas
