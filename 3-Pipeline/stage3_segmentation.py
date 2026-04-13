# =============================================================================
# stage3_segmentation.py  —  Valley Analysis, Blob Fusion & Word Grouping
#
# Handles the character-level segmentation based on the rough skeleton:
#   vertical projection → adaptive valley analysis → blob detection →
#   per-word fusion → final character segments + word groups
#
# INPUT  (from Stage 2 - Preprocessing):
#   - stems        : list[str]             — image stems to process
#   - work_root    : str                   — base working directory
#   - cfg          : PipelineConfig        — tunable parameters
#
# OUTPUT (to Stage 4 - Classification):
#   - list[str]    — image stems successfully segmented
#   - Updated meta.json with:
#       segments_path       : str   — path to final character segments JSON
#       word_segments_path  : str   — path to word groups JSON (if enabled)
#       n_segments          : int   — total character segment count
#       n_words             : int   — word count
#       valley_min_width    : int
#       word_gap_px         : float — adaptive threshold used (informational)
#       seg_method          : str   — per-word method summary e.g. "valley:2,blob:1"
#
# Segment JSON format  (list of [x_start, x_end] pairs — unchanged for Stage 4)
# Word group JSON format  ({word_index, seg_indices, x_start, x_end} — unchanged)
# =============================================================================

from __future__ import annotations

import os
import json
import statistics
from typing import Optional

import cv2
import numpy as np

from stage1_config import PipelineConfig

# =============================================================================
# VALLEY ANALYSIS
# =============================================================================

def _find_valley_segments(skeleton_canvas: np.ndarray,
                           cfg: PipelineConfig) -> tuple[list, list]:
    """
    Find character-level segments via vertical projection valleys.

    Returns
    -------
    segments  : list of [x_start, x_end]
    gap_widths: list of int — pixel width of each inter-segment gap, in order.
                len(gap_widths) == len(segments) - 1
    """
    projection = (skeleton_canvas > 0).sum(axis=0).astype(int)
    W          = projection.shape[0]
    in_seg     = False
    seg_start  = 0
    segments   = []
    gap_widths = []

    x = 0
    while x < W:
        if projection[x] > 0 and not in_seg:
            seg_start = x
            in_seg    = True
            x += 1
        elif projection[x] == 0 and in_seg:
            gap_start = x
            while x < W and projection[x] == 0:
                x += 1
            gap_w = x - gap_start
            if gap_w >= cfg.valley_min_width:
                segments.append([seg_start, gap_start])
                gap_widths.append(gap_w)
                in_seg = False
            # else: gap too narrow — stay in current segment
        else:
            x += 1

    if in_seg:
        segments.append([seg_start, W])

    return segments, gap_widths


def _adaptive_word_groups(segments: list,
                           gap_widths: list) -> tuple[list, float]:
    """
    Determine word boundaries from gap_widths using a statistical threshold.

    Any gap wider than  mean + 0.5 * stdev  is treated as a word boundary.
    Falls back gracefully when there are too few gaps to compute statistics.

    Returns
    -------
    word_groups     : list of word-group dicts (same schema Stage 4 expects)
    threshold_used  : float — the pixel threshold applied
    """
    if not segments:
        return [], 0.0

    if len(gap_widths) == 0:
        return [{
            "word_index":  0,
            "seg_indices": [0],
            "x_start":     segments[0][0],
            "x_end":       segments[0][1],
        }], 0.0

    mean_gap  = statistics.mean(gap_widths)
    stdev     = statistics.stdev(gap_widths) if len(gap_widths) > 1 else 0.0
    threshold = mean_gap + 0.5 * stdev

    word_groups  = []
    current_word = [0]

    for seg_i in range(1, len(segments)):
        gap_w = gap_widths[seg_i - 1]
        if gap_w >= threshold:
            word_groups.append({
                "word_index":  len(word_groups),
                "seg_indices": current_word[:],
                "x_start":     segments[current_word[0]][0],
                "x_end":       segments[current_word[-1]][1],
            })
            current_word = [seg_i]
        else:
            current_word.append(seg_i)

    word_groups.append({
        "word_index":  len(word_groups),
        "seg_indices": current_word[:],
        "x_start":     segments[current_word[0]][0],
        "x_end":       segments[current_word[-1]][1],
    })

    return word_groups, threshold

# =============================================================================
# BLOB DETECTION  (vertical splits only)
# =============================================================================

def _blob_segments(binary: np.ndarray, cfg: PipelineConfig) -> list:
    """
    Detect character blobs using connected components, enforcing that only
    horizontal (left-right) separations are respected.

    Any two blobs whose y-ranges overlap are considered part of the same
    vertical stack (e.g. a base character + its vowel diacritic above/below)
    and are merged into one combined x-span before producing segment boundaries.
    Only after all vertical merges are stable are x-axis boundaries extracted.

    Returns list of [x_start, x_end] pairs, sorted left-to-right.
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Collect bounding boxes; skip background (label 0) and tiny noise
    blobs = []
    for lbl in range(1, num_labels):
        x    = stats[lbl, cv2.CC_STAT_LEFT]
        y    = stats[lbl, cv2.CC_STAT_TOP]
        w    = stats[lbl, cv2.CC_STAT_WIDTH]
        h    = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < cfg.blob_min_area:
            continue
        blobs.append([x, y, x + w, y + h])   # [x1, y1, x2, y2]

    if not blobs:
        return []

    # ── Iteratively merge blobs that share any vertical overlap ──────────────
    changed = True
    while changed:
        changed = False
        merged  = []
        used    = [False] * len(blobs)
        for i in range(len(blobs)):
            if used[i]:
                continue
            bx1, by1, bx2, by2 = blobs[i]
            for j in range(i + 1, len(blobs)):
                if used[j]:
                    continue
                jx1, jy1, jx2, jy2 = blobs[j]
                # Vertical overlap: y-ranges [by1, by2) and [jy1, jy2) intersect
                if by1 < jy2 and jy1 < by2:
                    bx1     = min(bx1, jx1)
                    by1     = min(by1, jy1)
                    bx2     = max(bx2, jx2)
                    by2     = max(by2, jy2)
                    used[j] = True
                    changed = True
            merged.append([bx1, by1, bx2, by2])
            used[i] = True
        blobs = merged

    # ── Convert to [x_start, x_end], sort left-to-right ─────────────────────
    segments = sorted([[b[0], b[2]] for b in blobs], key=lambda s: s[0])

    # ── Merge any x-overlapping segments produced by the vertical merges ─────
    merged_segs = [segments[0]]
    for seg in segments[1:]:
        if seg[0] < merged_segs[-1][1]:   # x-overlap
            merged_segs[-1][1] = max(merged_segs[-1][1], seg[1])
        else:
            merged_segs.append(seg)

    return merged_segs

# =============================================================================
# PER-WORD FUSION  (valley vs blob)
# =============================================================================

def _clip_segs_to_word(segments: list, x_start: int, x_end: int) -> list:
    """Return only the segments that fall within [x_start, x_end]."""
    return [s for s in segments if s[0] >= x_start and s[1] <= x_end]


def _fuse_segments_for_word(valley_segs: list, blob_segs: list) -> tuple[list, str]:
    """
    Choose the better segment list for a single word's x-extent.

    - blob count > valley count  →  touching characters detected; use blob.
    - otherwise                  →  valley is at least as fine; use valley.
    """
    if not valley_segs:
        return blob_segs, "blob"
    if not blob_segs:
        return valley_segs, "valley"
    if len(blob_segs) > len(valley_segs):
        return blob_segs, "blob"
    return valley_segs, "valley"


def _build_fused_segments(valley_segs: list,
                           valley_word_groups: list,
                           blob_segs: list) -> tuple[list, list, str]:
    """
    Build the final per-image character segments and word groups by fusing
    valley and blob results at the word level.

    Returns
    -------
    final_segments  : list of [x_start, x_end]   (new flat segment list)
    final_words     : list of word-group dicts    (seg_indices updated to new flat indices)
    method_summary  : str  e.g. "valley:2,blob:1"
    """
    final_segments = []
    final_words    = []
    method_counts  = {"valley": 0, "blob": 0}

    for wg in valley_word_groups:
        w_x_start = wg["x_start"]
        w_x_end   = wg["x_end"]

        v_segs = _clip_segs_to_word(valley_segs, w_x_start, w_x_end)
        b_segs = _clip_segs_to_word(blob_segs,   w_x_start, w_x_end)

        chosen, method = _fuse_segments_for_word(v_segs, b_segs)
        method_counts[method] += 1

        seg_offset  = len(final_segments)
        seg_indices = list(range(seg_offset, seg_offset + len(chosen)))
        final_segments.extend(chosen)
        final_words.append({
            "word_index":  wg["word_index"],
            "seg_indices": seg_indices,
            "x_start":     w_x_start,
            "x_end":       w_x_end,
        })

    summary = ",".join(f"{k}:{v}" for k, v in method_counts.items() if v > 0)
    return final_segments, final_words, summary

# =============================================================================
# CROP HELPER  (used by Stage 4)
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
    if roi.shape[1] == 0:
        return None
    rh, rw = roi.shape
    if rh == 0 or rw == 0:
        return None
    if rh > cs or rw > cs:
        scale = min(cs / rw, cs / rh) * 0.9
        roi   = cv2.resize(roi,
                           (max(1, int(rw * scale)), max(1, int(rh * scale))),
                           interpolation=cv2.INTER_NEAREST)
        rh, rw = roi.shape
    canvas        = np.zeros((cs, cs), dtype=np.uint8)
    off_y         = (cs - rh) // 2
    off_x         = (cs - rw) // 2
    canvas[off_y:off_y + rh, off_x:off_x + rw] = roi
    return cv2.bitwise_not(canvas)

# =============================================================================
# PROCESS ONE IMAGE
# =============================================================================

def _process_one(stem: str,
                 work_root: str,
                 cfg: PipelineConfig) -> tuple[bool, int, int, str]:
    temp_dir  = os.path.join(work_root, "temp", stem)
    meta_path = os.path.join(temp_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # ── Load rough skeleton (inverted to white-ink-on-black) for valley analysis
    skel_path       = meta.get("skel_path")
    skeleton_img    = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
    if skeleton_img is None:
        return False, 0, 0, ""
    skeleton_canvas = cv2.bitwise_not(skeleton_img)   # white ink on black

    # ── Load binary (white ink on black) for blob detection ──────────────────
    binary = cv2.imread(meta["binary_path"], cv2.IMREAD_GRAYSCALE)
    if binary is None:
        return False, 0, 0, ""

    # ─────────────────────────────────────────────────────────────────────────
    # VALLEY ANALYSIS  (segments + gap widths)
    # ─────────────────────────────────────────────────────────────────────────
    valley_segs, gap_widths = _find_valley_segments(skeleton_canvas, cfg)

    # ─────────────────────────────────────────────────────────────────────────
    # ADAPTIVE WORD GROUPING  (from valley gap distribution)
    # ─────────────────────────────────────────────────────────────────────────
    threshold_used = 0.0
    if cfg.word_spacer_enabled and valley_segs:
        valley_word_groups, threshold_used = _adaptive_word_groups(
            valley_segs, gap_widths
        )
    else:
        valley_word_groups = [{
            "word_index":  0,
            "seg_indices": list(range(len(valley_segs))),
            "x_start":     valley_segs[0][0] if valley_segs else 0,
            "x_end":       valley_segs[-1][1] if valley_segs else skeleton_canvas.shape[1],
        }] if valley_segs else []

    # ─────────────────────────────────────────────────────────────────────────
    # BLOB DETECTION  (vertical-merge enforced, horizontal splits only)
    # ─────────────────────────────────────────────────────────────────────────
    blob_segs = _blob_segments(binary, cfg)

    # ─────────────────────────────────────────────────────────────────────────
    # PER-WORD FUSION
    # ─────────────────────────────────────────────────────────────────────────
    if valley_word_groups:
        final_segs, final_words, method_summary = _build_fused_segments(
            valley_segs, valley_word_groups, blob_segs
        )
    else:
        # No valley segments at all — fall back to pure blob result
        final_segs  = blob_segs
        final_words = [{
            "word_index":  0,
            "seg_indices": list(range(len(blob_segs))),
            "x_start":     blob_segs[0][0] if blob_segs else 0,
            "x_end":       blob_segs[-1][1] if blob_segs else skeleton_canvas.shape[1],
        }] if blob_segs else []
        method_summary = "blob:1"

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE OUTPUTS  (format unchanged — Stage 4 compatibility preserved)
    # ─────────────────────────────────────────────────────────────────────────
    segments_path = os.path.join(temp_dir, "segments.json")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(final_segs, f)

    word_segments_path = None
    n_words            = 0
    if cfg.word_spacer_enabled and final_words:
        word_segments_path = os.path.join(temp_dir, "word_segments.json")
        with open(word_segments_path, "w", encoding="utf-8") as f:
            json.dump(final_words, f, ensure_ascii=False, indent=2)
        n_words = len(final_words)

    # ─────────────────────────────────────────────────────────────────────────
    # UPDATE META
    # ─────────────────────────────────────────────────────────────────────────
    meta["segments_path"]      = segments_path
    meta["word_segments_path"] = word_segments_path
    meta["n_segments"]         = len(final_segs)
    meta["n_words"]            = n_words
    meta["valley_min_width"]   = cfg.valley_min_width
    meta["word_gap_px"]        = round(threshold_used, 2)   # adaptive, informational
    meta["seg_method"]         = method_summary

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return True, len(final_segs), n_words, method_summary

# =============================================================================
# RUN STAGE 3 — SEGMENTATION
# =============================================================================

def run_stage3_segmentation(cfg: PipelineConfig,
                             stems: list,
                             work_root: str) -> list[str]:
    print(f"  [Stage 3 - Segmentation] Analysing {len(stems)} image(s)  "
          f"[valley + blob fusion]...")
    if cfg.word_spacer_enabled:
        print(f"  [Stage 3 - Segmentation] Word spacer : ENABLED  (adaptive threshold)")
    else:
        print(f"  [Stage 3 - Segmentation] Word spacer : DISABLED")

    success_stems = []
    for i, stem in enumerate(stems, 1):
        try:
            ok, n_segs, n_words, method = _process_one(stem, work_root, cfg)
            if ok:
                success_stems.append(stem)
                info = f"  words: {n_words}" if cfg.word_spacer_enabled else ""
                print(f"    [{i}/{len(stems)}] {stem}  segs: {n_segs}{info}  [{method}]")
        except Exception as exc:
            import traceback
            print(f"    [{i}/{len(stems)}] {stem}  ERROR: {exc}")
            traceback.print_exc()

    print(f"  [Stage 3 - Segmentation] Done.\n")
    return success_stems