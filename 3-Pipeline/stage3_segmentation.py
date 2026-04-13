# =============================================================================
# stage3_segmentation.py  —  Valley Analysis & Word Grouping
#
# Handles the character-level segmentation based on the rough skeleton:
#   vertical projection → valley identification → word gap grouping
#
# INPUT:
#   - stems        : list[str]             — image stems to process
#   - work_root    : str                   — base working directory
#   - cfg          : PipelineConfig        — tunable parameters
#
# OUTPUT:
#   - list[str]    — image stems successfully segmented
#   - Updated meta.json with segments_path, n_segments, etc.
# =============================================================================

from __future__ import annotations

import os
import json
from typing import Optional

import cv2
import numpy as np

from stage1_config import PipelineConfig

# =============================================================================
# SEGMENTATION LOGIC
# =============================================================================

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
# CROP HELPER (Used by Stage 4)
# =============================================================================

def _make_window_crop_np(skeleton: np.ndarray,
                          x_start: int,
                          x_end: int,
                          cfg: PipelineConfig) -> Optional[np.ndarray]:
    """Extract and pad a skeleton crop for model input."""
    cs  = cfg.char_canvas_size
    pad = cfg.window_pad
    x1  = max(0, x_start - pad)
    x2  = min(skeleton.shape[1], x_end + pad)
    roi = skeleton[:, x1:x2]
    if roi.shape[1] == 0: return None
    rh, rw = roi.shape
    if rh == 0 or rw == 0: return None
    if rh > cs or rw > cs:
        scale = min(cs / rw, cs / rh) * 0.9
        roi = cv2.resize(roi, (max(1, int(rw * scale)), max(1, int(rh * scale))), interpolation=cv2.INTER_NEAREST)
        rh, rw = roi.shape
    canvas = np.zeros((cs, cs), dtype=np.uint8)
    off_y = (cs - rh) // 2
    off_x = (cs - rw) // 2
    canvas[off_y:off_y + rh, off_x:off_x + rw] = roi
    return cv2.bitwise_not(canvas)

# =============================================================================
# PROCESS ONE IMAGE
# =============================================================================

def _process_one(stem: str,
                 work_root: str,
                 cfg: PipelineConfig) -> tuple[bool, int, int]:
    temp_dir = os.path.join(work_root, "temp", stem)
    meta_path = os.path.join(temp_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # STRICT LOGIC RESTORATION: Use the rough skeleton (skel_path) for valley analysis
    # This precisely matches the original Stage 2 behavior.
    skel_path = meta.get("skel_path")
    skeleton_img = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
    if skeleton_img is None: return False, 0, 0
    
    # Invert back to white-on-black for projection
    skeleton_canvas = cv2.bitwise_not(skeleton_img)

    # Valley Segmentation
    segments = _find_valley_segments(skeleton_canvas, cfg)
    segments_path = os.path.join(temp_dir, "segments.json")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f)

    # Word Grouping
    word_segments_path = None
    n_words = 0
    if cfg.word_spacer_enabled:
        word_groups = _find_word_groups(segments, skeleton_canvas, cfg)
        word_segments_path = os.path.join(temp_dir, "word_segments.json")
        with open(word_segments_path, "w", encoding="utf-8") as f:
            json.dump(word_groups, f, ensure_ascii=False, indent=2)
        n_words = len(word_groups)

    # Update meta
    meta["segments_path"] = segments_path
    meta["word_segments_path"] = word_segments_path
    meta["n_segments"] = len(segments)
    meta["n_words"] = n_words

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return True, len(segments), n_words

# =============================================================================
# RUN STAGE 3 — SEGMENTATION
# =============================================================================

def run_stage3_segmentation(cfg: PipelineConfig,
                             stems: list,
                             work_root: str) -> list[str]:
    print(f"  [Stage 3 - Segmentation] Analysing valleys for {len(stems)} image(s)...")
    if cfg.word_spacer_enabled:
        print(f"  [Stage 3 - Segmentation] Word spacer : ENABLED  (gap >= {cfg.word_gap_px} px)")
    else:
        print(f"  [Stage 3 - Segmentation] Word spacer : DISABLED")

    success_stems = []
    for i, stem in enumerate(stems, 1):
        try:
            ok, n_segs, n_words = _process_one(stem, work_root, cfg)
            if ok:
                success_stems.append(stem)
                info = f"  words: {n_words}" if cfg.word_spacer_enabled else ""
                print(f"    [{i}/{len(stems)}] {stem}  segs: {n_segs}{info}")
        except Exception as exc:
            print(f"    [{i}/{len(stems)}] {stem}  ERROR: {exc}")

    print(f"  [Stage 3 - Segmentation] Done.\n")
    return success_stems
