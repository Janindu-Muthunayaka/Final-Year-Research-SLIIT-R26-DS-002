# =============================================================================
# stage4_classification.py  —  Model Loading, GPU Inference & Recognition
#
# Handles the EfficientNetV2-S classification pipeline:
#   batched GPU inference → variant-map recognition → word annotation → Metrics
#
# INPUT  (from Stage 3 - Segmentation):
#   - stems        : list[str]             — image stems to process
#   - work_root    : str                   — base working directory
#   - model        : nn.Module             — loaded EfficientNetV2-S
#   - device       : torch.device          — CUDA or CPU
#   - idx_to_class : dict[str, str]        — {index: class_name}
#   - cfg          : PipelineConfig        — tunable parameters
#
# OUTPUT (to stage5_reporting):
#   - list[dict]  — per-image result dicts
#
# Recognition logic — Variant-Map Recognition (replaces greedy multi-seg):
# ─────────────────────────────────────────────────────────────────────────
#   For each segment position `pos` in the sentence:
#
#   1. Predict the single-segment crop at `pos`  → base_char, base_conf
#
#   2. Determine word boundary context from Stage 3 word groups:
#        - has_left  : pos-1 exists AND is in the same word
#        - has_right : pos+1 exists AND is in the same word
#        (Special 2-back rule: only ම and ව trigger look-back of 2 segments)
#
#   3. Build candidate crops (respecting word boundaries + overlap rules):
#        - crop_left       : segments[pos-1..pos]   (if has_left)
#        - crop_right      : segments[pos..pos+1]   (if has_right)
#        - crop_both       : segments[pos-1..pos+1] (if has_left AND has_right)
#
#   4. Batch-predict all non-None candidate crops.
#
#   5. Validate candidates against VARIANT_MAP:
#        For each candidate crop, its top-1 predicted class must appear as a
#        VALUE under the KEY that equals `base_char` in VARIANT_MAP.
#        Candidates that are not in VARIANT_MAP[base_char] are discarded.
#
#   6. Priority among validated candidates:
#        a. If base_char itself is a non-key (already a full akshara with
#           diacritics — i.e. NOT a bare consonant key in VARIANT_MAP):
#              → HIGHEST priority; still test left/right for completeness
#                but this standalone prediction wins if its confidence ≥
#                all validated compound candidates.
#        b. Among compound candidates: both > right > left
#        c. If confidence tie: both > right > left (priority breaks tie)
#
#   7. Overlap protection: once a compound (left, right, or both) is
#      accepted, the consumed neighbour positions are marked "used" and
#      skipped in future iterations.
#
#   8. If no validated candidate beats the standalone prediction (or if
#      base_char is not a key in VARIANT_MAP), emit base_char alone.
# =============================================================================

from __future__ import annotations

import os
import json
import concurrent.futures
from typing import Optional

import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
from torch.amp import autocast

from stage1_config import (
    PipelineConfig,
    _NORM_MEAN, _NORM_STD,
    INFER_BATCH_SIZE, P_CHAR_CANVAS_SIZE, TOP_K, PNG_POOL_WORKERS,
)
from stage3_segmentation import _make_window_crop_np

# =============================================================================
# VARIANT MAP  — loaded once at import time
# =============================================================================

def _load_variant_map(variants_path: str) -> dict[str, set[str]]:
    """
    Load VARIANT_MAP from Variants.py and return it as
    { base_char : set_of_valid_compound_strings }.

    The values are stored as a set for O(1) membership checks.
    """
    import importlib.util, sys
    spec   = importlib.util.spec_from_file_location("_variants_mod", variants_path)
    mod    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw: dict[str, list[str]] = mod.VARIANT_MAP
    return {k: set(v) for k, v in raw.items()}


# Resolved path — users may also pass cfg.variants_path; we fall back to the
# sibling file "Variants.py" located next to this script.
_VARIANTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Variants.py")
_VARIANT_MAP: dict[str, set[str]] = {}

def _ensure_variant_map(variants_path: Optional[str] = None) -> dict[str, set[str]]:
    """Return the module-level VARIANT_MAP, loading it lazily on first call."""
    global _VARIANT_MAP
    if not _VARIANT_MAP:
        path = variants_path or _VARIANTS_FILE
        if os.path.isfile(path):
            _VARIANT_MAP = _load_variant_map(path)
            print(f"  [Stage 4 – Classification] Variant map loaded  "
                  f"({len(_VARIANT_MAP)} base chars)  ← {path}")
        else:
            print(f"  [Stage 4 – Classification] WARNING: Variants.py not found at {path}. "
                  f"Variant-map recognition disabled — falling back to single-seg only.")
    return _VARIANT_MAP

# Characters whose bare form is itself a full akshara (already carries a
# diacritic). These are the *values* across all VARIANT_MAP entries — i.e.
# they are NOT keys at the top level.  We compute this set dynamically after
# the map is loaded so it stays in sync with the actual file.
def _build_non_key_set(vmap: dict[str, set[str]]) -> set[str]:
    """
    Returns the set of characters that appear only as compound VALUES, never
    as bare consonant KEYS.  A prediction of one of these means the model
    already identified a pre-formed akshara and we give it the highest priority.
    """
    all_values: set[str] = set()
    for variants in vmap.values():
        all_values.update(variants)
    return all_values - set(vmap.keys())

# Two special consonants that may look 2 segments back
_TWO_BACK_CHARS: frozenset[str] = frozenset({"ම", "ව"})

# =============================================================================
# PNG WRITE POOL  (I/O off the main thread)
# =============================================================================

_PNG_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=PNG_POOL_WORKERS)

def _write_png_blocking(numpy_img: np.ndarray, path: str) -> None:
    _, buf = cv2.imencode(".png", numpy_img)
    with open(path, "wb") as fh:
        fh.write(buf)

def _save_png_async(numpy_img: np.ndarray,
                    path: str) -> concurrent.futures.Future:
    return _PNG_POOL.submit(_write_png_blocking, numpy_img.copy(), path)

# =============================================================================
# HELPERS
# =============================================================================

def _class_to_sinhala(class_name: str) -> str:
    """Extract the Sinhala character from a class name like 'vowel_අ_0001'."""
    parts = class_name.split("_")
    return parts[1] if len(parts) >= 3 else class_name

# =============================================================================
# MODEL LOADING
# =============================================================================

def _load_model(model_path: str, num_classes: int, device: torch.device):
    m = models.efficientnet_v2_s(weights=None)
    in_features       = m.classifier[1].in_features
    m.classifier[1]   = nn.Linear(in_features, num_classes)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device).eval()

    val_acc = ckpt.get("val_acc", 0)
    if isinstance(val_acc, torch.Tensor):
        val_acc = val_acc.item()

    print(f"  [Stage 4 – Classification] Model     : EfficientNetV2-S  ({num_classes} classes)")
    print(f"  [Stage 4 – Classification] Device    : {device}")
    print(f"  [Stage 4 – Classification] Checkpoint: epoch {ckpt.get('epoch', '?')} | "
          f"val acc {val_acc * 100:.2f}%")

    if device.type == "cuda":
        _warmup_model(m, device, num_classes)
    return m

def _warmup_model(model, device: torch.device, num_classes: int) -> None:
    cs = P_CHAR_CANVAS_SIZE
    try:
        dummy = torch.zeros(1, 3, cs, cs, device=device)
        with torch.no_grad():
            with autocast(device_type="cuda"):
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"  [Stage 4 – Classification] cuDNN warmup complete  (batch size = 1, {cs}×{cs})")
    except Exception as exc:
        print(f"  [Stage 4 – Classification] cuDNN warmup skipped: {exc}")

# =============================================================================
# NUMPY → PINNED TENSOR
# =============================================================================

def _np_to_tensor_pinned(gray_np: np.ndarray,
                          device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(gray_np).float().div_(255.0)
    t = t.unsqueeze(0).expand(3, -1, -1).contiguous()
    if device.type == "cuda":
        t = t.pin_memory()
        t = t.to(device, non_blocking=True)
    else:
        t = t.to(device)
    mean = _NORM_MEAN.to(device)
    std  = _NORM_STD.to(device)
    t    = t.sub_(mean[:, None, None]).div_(std[:, None, None])
    return t.unsqueeze(0)

# =============================================================================
# BATCHED INFERENCE
# =============================================================================

def _predict_batch(crops_np: list,
                   model,
                   device: torch.device,
                   idx_to_class: dict) -> list:
    """
    Predict a list of crops (some may be None).
    Returns a parallel list of predictions or None where the crop was None.
    Each prediction is a list of (class_name, confidence_pct) tuples, length TOP_K.
    """
    valid_idx = [i for i, c in enumerate(crops_np) if c is not None]
    if not valid_idx:
        return [None] * len(crops_np)
    all_preds = [None] * len(crops_np)
    for chunk_start in range(0, len(valid_idx), INFER_BATCH_SIZE):
        chunk = valid_idx[chunk_start: chunk_start + INFER_BATCH_SIZE]
        tensors = torch.cat(
            [_np_to_tensor_pinned(crops_np[i], device) for i in chunk], dim=0
        )
        with torch.no_grad():
            if device.type == "cuda":
                with autocast(device_type="cuda"):
                    logits = model(tensors)
                    probs_all = torch.softmax(logits.float(), dim=1)
            else:
                logits    = model(tensors)
                probs_all = torch.softmax(logits, dim=1)
            top_probs, top_i = torch.topk(probs_all, k=TOP_K)
        top_probs_cpu = top_probs.cpu()
        top_i_cpu     = top_i.cpu()
        for batch_pos, orig_idx in enumerate(chunk):
            all_preds[orig_idx] = [
                (idx_to_class[str(top_i_cpu[batch_pos, k].item())],
                 round(top_probs_cpu[batch_pos, k].item() * 100, 2))
                for k in range(TOP_K)
            ]
    return all_preds

# =============================================================================
# WORD-BOUNDARY HELPER
# =============================================================================

def _build_seg_to_word(word_groups: list) -> dict[int, int]:
    """Return {seg_index: word_index} from the Stage 3 word-group list."""
    mapping: dict[int, int] = {}
    for wg in word_groups:
        for si in wg["seg_indices"]:
            mapping[si] = wg["word_index"]
    return mapping

# =============================================================================
# VARIANT-MAP RECOGNITION  (replaces _greedy_segment entirely)
# =============================================================================

def _variant_map_segment(skeleton:    np.ndarray,
                          segments:   list,
                          word_groups: list,
                          model,
                          device:      torch.device,
                          idx_to_class: dict,
                          out_dir:     str,
                          cfg:         PipelineConfig,
                          vmap:        dict[str, set[str]],
                          non_keys:    set[str]) -> list:
    """
    Variant-map-driven recognition loop.

    Parameters
    ----------
    skeleton    : white-ink-on-black MAT skeleton array
    segments    : list of [x_start, x_end] from Stage 3
    word_groups : list of word-group dicts from Stage 3
    model       : EfficientNetV2-S (eval mode)
    device      : torch.device
    idx_to_class: {str_index: class_name}
    out_dir     : directory to write akshara PNG crops
    cfg         : PipelineConfig
    vmap        : VARIANT_MAP as {base_char: set_of_compound_strings}
    non_keys    : chars that appear only as values, never as bare-consonant keys

    Returns
    -------
    list of akshara result dicts — same schema as the old _greedy_segment so
    downstream code (_annotate_word_indices, _process_one) is unchanged.
    """
    N            = len(segments)
    seg_to_word  = _build_seg_to_word(word_groups) if word_groups else {}
    used         = [False] * N      # segments consumed by a compound crop
    results      = []
    akshara_idx  = 0
    png_futures  = []

    # ── helper: are two adjacent positions in the same word? ─────────────────
    def _same_word(a: int, b: int) -> bool:
        if not word_groups:
            return True                             # no word info → treat as one word
        if a < 0 or b >= N:
            return False
        return seg_to_word.get(a, -1) == seg_to_word.get(b, -2)

    pos = 0
    while pos < N:
        if used[pos]:
            pos += 1
            continue

        # ── Step 1: predict the single segment at pos ─────────────────────────
        crop_base = _make_window_crop_np(skeleton,
                                         segments[pos][0], segments[pos][1], cfg)
        if crop_base is None:
            pos += 1
            continue

        [preds_base] = _predict_batch([crop_base], model, device, idx_to_class)
        if preds_base is None:
            pos += 1
            continue

        base_char = _class_to_sinhala(preds_base[0][0])
        base_conf = preds_base[0][1]

        # ── Step 2: determine valid neighbour positions ───────────────────────
        # User constraint: If standalone confidence is >= 95%, it cannot be 
        # "given away" to a neighbor (as a left-side diacritic/kombuwa).
        STRICT_THRESHOLD = 95.0
        is_locked        = (base_conf >= STRICT_THRESHOLD)

        if is_locked:
            # Prevent LOOKING BACK (to avoid double-grabbing)
            has_left = False
            # We still allow LOOKING RIGHT, but we'll restrict what it can find
            # in the validation step (Step 5).
        else:
            # Left look-back is 2 positions for ම / ව, else 1
            left_reach = 2 if base_char in _TWO_BACK_CHARS else 1

            # Collect candidate left positions (skip already-used segments)
            left_positions: list[int] = []
            for delta in range(1, left_reach + 1):
                lp = pos - delta
                if lp >= 0 and not used[lp] and _same_word(lp, pos):
                    left_positions.append(lp)
                else:
                    break    # stop at the first invalid step

            has_left  = bool(left_positions)
            left_pos  = left_positions[0] if has_left else None   # immediate left
            far_left  = left_positions[-1] if len(left_positions) == 2 else left_pos

        # Both modes (locked or not) check the next segment for right-side variants
        has_right = (pos + 1 < N and not used[pos + 1]
                    and _same_word(pos, pos + 1))
        
        right_pos = pos + 1 if has_right else None

        # ── Step 3: build candidate crops ─────────────────────────────────────
        # crop_left  : far_left..pos    (spans 1 or 2 segments to the left)
        # crop_right : pos..right_pos
        # crop_both  : far_left..right_pos
        crop_left  = (_make_window_crop_np(skeleton,
                                            segments[far_left][0],
                                            segments[pos][1], cfg)
                      if has_left else None)
        crop_right = (_make_window_crop_np(skeleton,
                                            segments[pos][0],
                                            segments[right_pos][1], cfg)
                      if has_right else None)
        crop_both  = (_make_window_crop_np(skeleton,
                                            segments[far_left][0],
                                            segments[right_pos][1], cfg)
                      if (has_left and has_right) else None)

        # ── Step 4: batch-predict all compound candidates ─────────────────────
        compound_crops = [crop_left, crop_right, crop_both]
        compound_preds = _predict_batch(compound_crops, model, device, idx_to_class)
        preds_left, preds_right, preds_both = compound_preds

        # ── Step 5: validate compounds against VARIANT_MAP ───────────────────
        valid_variants = vmap.get(base_char, set())  # empty set if base_char not a key

        def _validated(preds, is_right_merge: bool = False) -> Optional[tuple[str, float]]:
            """Return (predicted_char, conf) if top-1 is in VARIANT_MAP, else None."""
            if preds is None:
                return None
            char = _class_to_sinhala(preds[0][0])
            conf = preds[0][1]

            # Locked logic: 
            # If we are locked and looking right, we ONLY accept merges that are
            # valid variants of the CURRENT character (right-side diacritics).
            # We reject merges that would result in a foreign character (non_keys).
            if is_locked and is_right_merge:
                if char in valid_variants:
                    return char, conf
                return None

            # Normal logic (no lock):
            # Valid if it's a known variant of the current base OR any known variant at all
            if char in valid_variants or char in non_keys:
                return char, conf
            return None

        v_left  = _validated(preds_left)
        v_right = _validated(preds_right, is_right_merge=True)
        v_both  = _validated(preds_both, is_right_merge=True)

        # ── Step 6: choose the winner ─────────────────────────────────────────
        #
        # Priority ordering:
        #   a) base_char is a non-key (already a full akshara):
        #      it wins unless a compound has STRICTLY higher confidence.
        #   b) among compounds: both > right > left
        #   c) if no compound validated, emit base_char alone.
        #
        # "is_non_key" means the model already returned a composed akshara
        # (e.g. "දා") without needing a compound crop; give it a head-start.

        is_non_key = base_char in non_keys

        # Build ordered candidate list: (char, conf, crop, label, segs_consumed)
        # segs_consumed is a list of positions that become used[] if chosen.
        candidates = []

        if v_both is not None:
            candidates.append(("both",  v_both[0],  v_both[1],
                                crop_both,  preds_both,
                                [far_left, pos, right_pos]
                                if far_left != left_pos
                                else [left_pos, pos, right_pos]))
        if v_right is not None:
            candidates.append(("right", v_right[0], v_right[1],
                                crop_right, preds_right,
                                [pos, right_pos]))
        if v_left is not None:
            candidates.append(("left",  v_left[0],  v_left[1],
                                crop_left,  preds_left,
                                [far_left, pos]
                                if far_left != left_pos
                                else [left_pos, pos]))

        # standalone baseline entry (always present)
        candidates.append(("base", base_char, base_conf,
                            crop_base, preds_base, [pos]))

        # In 'Strict Structure' mode, we use raw confidence.
        # We add a subtle 'Right-Side Override' to prioritize combinations 
        # that are direct variants of the base character (right/top/bottom diacritics).
        RIGHT_SIDE_BONUS = 2.0  

        def _score(entry) -> float:
            direction, char, conf, _, _, _ = entry
            score = conf
            # Only boost if it's a merge that 'belongs' to the current base
            if direction in ("right", "both") and char in valid_variants:
                score += RIGHT_SIDE_BONUS
            return score

        # Sort: highest adjusted-score first; ties broken by order above
        # (both > right > left > base via stable sort + index trick)
        priority = {"both": 0, "right": 1, "left": 2, "base": 3}
        candidates.sort(key=lambda e: (-_score(e), priority[e[0]]))

        winner = candidates[0]
        w_dir, w_char, w_conf, w_crop, w_preds, w_segs = winner

        # ── Step 7: mark consumed segments as used ────────────────────────────
        for si in w_segs:
            used[si] = True

        # Determine segment span for metadata
        seg_start_idx  = min(w_segs)
        seg_end_idx    = max(w_segs)
        n_segs_used    = len(w_segs)
        x_start        = segments[seg_start_idx][0]
        x_end          = segments[seg_end_idx][1]

        # ── Step 8: write crop PNG ────────────────────────────────────────────
        crop_path = os.path.join(out_dir, f"akshara_{akshara_idx:03d}.png")
        png_futures.append(_save_png_async(w_crop, crop_path))

        chosen_by = (
            f"variant-map:{w_dir} ({w_conf:.1f}%)"
            if w_dir != "base"
            else f"standalone{'[non-key]' if is_non_key else ''} ({w_conf:.1f}%)"
        )

        results.append({
            "index":          akshara_idx,
            "seg_start":      seg_start_idx,
            "seg_end":        seg_end_idx,
            "window_segs":    n_segs_used,
            "x_start":        x_start,
            "x_end":          x_end,
            "chosen_by":      chosen_by,
            "confidence":     w_conf,
            "crop_path":      crop_path,
            "predictions":    [[p[0], p[1]] for p in w_preds],
            "predicted_char": w_char,
            "word_index":     None,
        })
        akshara_idx += 1
        pos += 1  # always advance by 1; consumed neighbours are skipped via used[]

    concurrent.futures.wait(png_futures)
    return results

# =============================================================================
# Public alias kept for any external callers that import _greedy_segment
# =============================================================================

def _greedy_segment(skeleton, segments, model, device, idx_to_class,
                    out_dir, cfg, word_groups=None):
    """
    Compatibility shim — routes to the new variant-map recogniser.
    External code that imported _greedy_segment continues to work unchanged.
    """
    vmap     = _ensure_variant_map(getattr(cfg, "variants_path", None))
    non_keys = _build_non_key_set(vmap)
    wg       = word_groups or []
    return _variant_map_segment(skeleton, segments, wg, model, device,
                                 idx_to_class, out_dir, cfg, vmap, non_keys)

# =============================================================================
# WORD ANNOTATION & METRICS
# =============================================================================

def _annotate_word_indices(char_results: list, word_groups: list) -> str:
    if not word_groups:
        for ak in char_results: ak["word_index"] = 0
        return "".join(ak["predicted_char"] for ak in char_results)
    seg_to_word = {}
    for wg in word_groups:
        for si in wg["seg_indices"]: seg_to_word[si] = wg["word_index"]
    for ak in char_results:
        ak["word_index"] = seg_to_word.get(ak["seg_start"], 0)
    words_text         = []
    prev_word          = None
    current_word_chars = []
    for ak in char_results:
        wi = ak["word_index"]
        if prev_word is None: prev_word = wi
        if wi != prev_word:
            words_text.append("".join(current_word_chars))
            current_word_chars = []
            prev_word = wi
        current_word_chars.append(ak["predicted_char"])
    if current_word_chars: words_text.append("".join(current_word_chars))
    return " ".join(words_text)

def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp   = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp  = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev  = temp
    return dp[n]

def compute_cer(pred: str, ref: str) -> float:
    if not ref: return 0.0
    return round(_edit_distance(list(pred), list(ref)) / len(list(ref)) * 100, 2)

def compute_wer(pred: str, ref: str) -> float:
    pw, rw = pred.split(), ref.split()
    if not rw: return 0.0
    return round(_edit_distance(pw, rw) / len(rw) * 100, 2)

# =============================================================================
# PROCESS ONE IMAGE
# =============================================================================

def _process_one(stem: str,
                 work_root: str,
                 model,
                 device: torch.device,
                 idx_to_class: dict,
                 cfg: PipelineConfig) -> dict:
    temp_dir  = os.path.join(work_root, "temp", stem)
    meta_path = os.path.join(temp_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f: meta = json.load(f)

    ground_truth  = meta["ground_truth"]
    skeleton_path = meta.get("skel_refined_path") or meta.get("skel_path")
    skeleton      = cv2.bitwise_not(cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE))

    with open(meta["segments_path"], "r", encoding="utf-8") as f:
        segments = json.load(f)

    N = len(segments)
    if N == 0:
        results_data = {
            "stem": stem, "ground_truth": ground_truth,
            "predicted_text": "", "predicted_text_no_spaces": "",
            "wer": 100.0, "cer": 100.0,
            "word_spacer_enabled": cfg.word_spacer_enabled, "aksharas": [],
        }
    else:
        # ── Load word groups (always, so variant-map respects word boundaries) ─
        word_groups: list = []
        if cfg.word_spacer_enabled and meta.get("word_segments_path"):
            ws_path = meta["word_segments_path"]
            if os.path.exists(ws_path):
                with open(ws_path, "r", encoding="utf-8") as f:
                    word_groups = json.load(f)

        # ── Load variant map (once per process) ───────────────────────────────
        variants_path = getattr(cfg, "variants_path", None)
        vmap          = _ensure_variant_map(variants_path)
        non_keys      = _build_non_key_set(vmap)

        # ── Run variant-map recognition ───────────────────────────────────────
        char_results = _variant_map_segment(
            skeleton     = skeleton,
            segments     = segments,
            word_groups  = word_groups,
            model        = model,
            device       = device,
            idx_to_class = idx_to_class,
            out_dir      = temp_dir,
            cfg          = cfg,
            vmap         = vmap,
            non_keys     = non_keys,
        )

        predicted_text_plain = "".join(r["predicted_char"] for r in char_results)

        # ── Word annotation ───────────────────────────────────────────────────
        if cfg.word_spacer_enabled and word_groups:
            predicted_text = _annotate_word_indices(char_results, word_groups)
        else:
            predicted_text = predicted_text_plain
            for ak in char_results: ak["word_index"] = 0

        wer     = compute_wer(predicted_text, ground_truth) if ground_truth else 0.0
        cer     = compute_cer(predicted_text, ground_truth) if ground_truth else 0.0
        n_words = (max((ak["word_index"] or 0) for ak in char_results) + 1
                   if char_results else 0)

        results_data = {
            "stem":                    stem,
            "ground_truth":            ground_truth,
            "predicted_text":          predicted_text,
            "predicted_text_no_spaces": predicted_text_plain,
            "wer":                     wer,
            "cer":                     cer,
            "n_segments":              N,
            "n_aksharas":              len(char_results),
            "n_multi_seg":             sum(1 for r in char_results
                                          if r["window_segs"] > 1),
            "n_words":                 n_words if cfg.word_spacer_enabled else None,
            "multi_seg_threshold":     cfg.multi_seg_threshold,
            "word_spacer_enabled":     cfg.word_spacer_enabled,
            "aksharas":                char_results,
        }

    with open(os.path.join(temp_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    return results_data

# =============================================================================
# RUN STAGE 4 — CLASSIFICATION
# =============================================================================

def run_stage4_classification(cfg: PipelineConfig,
                               stems: list,
                               work_root: str,
                               model,
                               device: torch.device,
                               idx_to_class: dict) -> list:
    # Pre-load variant map here so the "loaded" message appears exactly once
    # in the log, before per-image processing begins.
    variants_path = getattr(cfg, "variants_path", None)
    _ensure_variant_map(variants_path)

    print(f"  [Stage 4 - Classification] Classifying {len(stems)} image(s) on {device}...\n")
    all_results = []
    for i, stem in enumerate(stems, 1):
        print(f"    [{i}/{len(stems)}]", end=" ")
        try:
            res = _process_one(stem, work_root, model, device, idx_to_class, cfg)
            all_results.append(res)
            print()
        except Exception as exc:
            import traceback
            print(f"\n           ERROR: {exc}")
            traceback.print_exc()
    if device.type == "cuda": torch.cuda.synchronize()
    print(f"  [Stage 4 - Classification] Done.\n")
    return all_results