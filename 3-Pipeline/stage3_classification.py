# =============================================================================
# stage3_classification.py  —  Model Loading, GPU Inference & Metrics
#
# Handles the EfficientNetV2-S classification pipeline:
#   skeleton → window crops → batched GPU inference → greedy segmentation
#   → word annotation → CER / WER computation
#
# INPUT  (from stage2_preprocessing via run_stage3_classification):
#   - stems        : list[str]             — image stems to process
#   - work_root    : str                   — base working directory
#   - model        : nn.Module             — loaded EfficientNetV2-S
#   - device       : torch.device          — CUDA or CPU
#   - idx_to_class : dict[str, str]        — {index: class_name}
#   - cfg          : PipelineConfig        — tunable parameters
#
# OUTPUT (from run_stage3_classification):
#   - list[dict]  — per-image result dicts, each containing:
#       stem                     : str        — image stem
#       ground_truth             : str        — label text
#       predicted_text           : str        — predicted text (with spaces)
#       predicted_text_no_spaces : str        — predicted text (no spaces)
#       wer                      : float      — Word Error Rate (%)
#       cer                      : float      — Character Error Rate (%)
#       n_segments               : int        — total valley segments
#       n_aksharas               : int        — recognised character count
#       n_multi_seg              : int        — multi-segment akshara count
#       n_words                  : int|None   — word count (if spacer enabled)
#       aksharas                 : list[dict] — per-character predictions
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

# =============================================================================
# PNG WRITE POOL  (I/O off the main thread)
# =============================================================================

_PNG_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=PNG_POOL_WORKERS)

# =============================================================================
# HELPERS
# =============================================================================

def _class_to_sinhala(class_name: str) -> str:
    """Extract the Sinhala character from a class name like 'vowel_අ_0001'."""
    parts = class_name.split("_")
    return parts[1] if len(parts) >= 3 else class_name


def _write_png_blocking(numpy_img: np.ndarray, path: str) -> None:
    _, buf = cv2.imencode(".png", numpy_img)
    with open(path, "wb") as fh:
        fh.write(buf)


def _save_png_async(numpy_img: np.ndarray,
                    path: str) -> concurrent.futures.Future:
    return _PNG_POOL.submit(_write_png_blocking, numpy_img.copy(), path)

# =============================================================================
# SKELETON REBUILD (from binary — used at classification time)
# =============================================================================

def _build_full_skeleton(binary: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Rebuild skeleton from binary image for classification window crops."""
    from skimage.morphology import skeletonize as sk_skeletonize
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (cfg.close_k, cfg.close_k))
    healed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    skel   = sk_skeletonize(healed > 0).astype(np.uint8) * 255
    if cfg.skeleton_dil > 0:
        skel = cv2.dilate(skel, np.ones((3, 3), np.uint8),
                          iterations=cfg.skeleton_dil)
    return skel

# =============================================================================
# WINDOW CROP
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
# MODEL LOADING
# =============================================================================

def _load_model(model_path: str, num_classes: int, device: torch.device):
    """
    Load EfficientNetV2-S onto `device`.
    On CUDA the model is wrapped in a no-grad eval mode.
    """
    m = models.efficientnet_v2_s(weights=None)
    in_features       = m.classifier[1].in_features
    m.classifier[1]   = nn.Linear(in_features, num_classes)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device).eval()

    val_acc = ckpt.get("val_acc", 0)
    if isinstance(val_acc, torch.Tensor):
        val_acc = val_acc.item()

    print(f"  [Stage 3 – Classification] Model     : EfficientNetV2-S  ({num_classes} classes)")
    print(f"  [Stage 3 – Classification] Device    : {device}")
    print(f"  [Stage 3 – Classification] Checkpoint: epoch {ckpt.get('epoch', '?')} | "
          f"val acc {val_acc * 100:.2f}%")

    if device.type == "cuda":
        _warmup_model(m, device, num_classes)

    return m


def _warmup_model(model, device: torch.device, num_classes: int) -> None:
    """Feed a dummy batch to trigger cuDNN kernel selection."""
    cs = P_CHAR_CANVAS_SIZE
    try:
        dummy = torch.zeros(1, 3, cs, cs, device=device)
        with torch.no_grad():
            with autocast(device_type="cuda"):
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"  [Stage 3 – Classification] cuDNN warmup complete  (batch size = 1, {cs}×{cs})")
    except Exception as exc:
        print(f"  [Stage 3 – Classification] cuDNN warmup skipped: {exc}")

# =============================================================================
# NUMPY → PINNED TENSOR
# =============================================================================

def _np_to_tensor_pinned(gray_np: np.ndarray,
                          device: torch.device) -> torch.Tensor:
    """
    Convert an uint8 grayscale numpy array to a normalised float tensor.
    On CUDA, pin_memory is used for async DMA copy.
    """
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
# BATCHED INFERENCE  (GPU-aware)
# =============================================================================

def _predict_batch(crops_np: list,
                   model,
                   device: torch.device,
                   idx_to_class: dict) -> list:
    """
    Run a forward pass for up to len(crops_np) crops.
    Crops are assembled into a single batch.  On CUDA, FP16 autocast is used.
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
# GREEDY SEGMENTATION
# =============================================================================

def _greedy_segment(skeleton: np.ndarray,
                    segments: list,
                    model,
                    device: torch.device,
                    idx_to_class: dict,
                    out_dir: str,
                    cfg: PipelineConfig) -> list:
    """Greedy multi-segment classification with 1/2/3 segment window."""
    N           = len(segments)
    pos         = 0
    results     = []
    akshara_idx = 0
    png_futures = []

    while pos < N:
        crop_1 = _make_window_crop_np(skeleton, segments[pos][0],
                                      segments[pos][1], cfg)
        crop_2 = (_make_window_crop_np(skeleton, segments[pos][0],
                                       segments[pos + 1][1], cfg)
                  if pos + 1 < N else None)
        crop_3 = (_make_window_crop_np(skeleton, segments[pos][0],
                                       segments[pos + 2][1], cfg)
                  if pos + 2 < N else None)

        preds_1, preds_2, preds_3 = _predict_batch(
            [crop_1, crop_2, crop_3], model, device, idx_to_class
        )

        best_multi      = None
        best_multi_conf = 0.0
        best_multi_segs = 0

        if preds_2 is not None:
            conf2 = preds_2[0][1]
            if conf2 >= cfg.multi_seg_threshold and conf2 > best_multi_conf:
                best_multi      = (crop_2, preds_2)
                best_multi_conf = conf2
                best_multi_segs = 2

        if preds_3 is not None:
            conf3 = preds_3[0][1]
            if conf3 >= cfg.multi_seg_threshold and conf3 > best_multi_conf:
                best_multi      = (crop_3, preds_3)
                best_multi_conf = conf3
                best_multi_segs = 3

        if best_multi is not None:
            chosen_crop, preds = best_multi
            n_segs    = best_multi_segs
            chosen_by = f"{n_segs}-seg ({preds[0][1]:.1f}%)"
            x_end_seg = pos + n_segs - 1
        else:
            if crop_1 is None or preds_1 is None:
                pos += 1
                continue
            chosen_crop = crop_1
            preds       = preds_1
            n_segs      = 1
            chosen_by   = f"1-seg fallback ({preds[0][1]:.1f}%)"
            x_end_seg   = pos

        crop_path = os.path.join(out_dir, f"akshara_{akshara_idx:03d}.png")
        png_futures.append(_save_png_async(chosen_crop, crop_path))

        results.append({
            "index":          akshara_idx,
            "seg_start":      pos,
            "seg_end":        x_end_seg,
            "window_segs":    n_segs,
            "x_start":        segments[pos][0],
            "x_end":          segments[x_end_seg][1],
            "chosen_by":      chosen_by,
            "confidence":     preds[0][1],
            "crop_path":      crop_path,
            "predictions":    [[p[0], p[1]] for p in preds],
            "predicted_char": _class_to_sinhala(preds[0][0]),
            "word_index":     None,
        })

        akshara_idx += 1
        pos += n_segs

    concurrent.futures.wait(png_futures)
    return results

# =============================================================================
# WORD ANNOTATION
# =============================================================================

def _annotate_word_indices(char_results: list, word_groups: list) -> str:
    """Assign word indices to aksharas and build space-separated predicted text."""
    if not word_groups:
        for ak in char_results:
            ak["word_index"] = 0
        return "".join(ak["predicted_char"] for ak in char_results)

    seg_to_word = {}
    for wg in word_groups:
        for si in wg["seg_indices"]:
            seg_to_word[si] = wg["word_index"]

    for ak in char_results:
        ak["word_index"] = seg_to_word.get(ak["seg_start"], 0)

    words_text         = []
    prev_word          = None
    current_word_chars = []
    for ak in char_results:
        wi = ak["word_index"]
        if prev_word is None:
            prev_word = wi
        if wi != prev_word:
            words_text.append("".join(current_word_chars))
            current_word_chars = []
            prev_word = wi
        current_word_chars.append(ak["predicted_char"])
    if current_word_chars:
        words_text.append("".join(current_word_chars))

    return " ".join(words_text)

# =============================================================================
# METRICS
# =============================================================================

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
    if not ref:
        return 0.0
    return round(_edit_distance(list(pred), list(ref)) / len(list(ref)) * 100, 2)


def compute_wer(pred: str, ref: str) -> float:
    pw, rw = pred.split(), ref.split()
    if not rw:
        return 0.0
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
    """Classify all segments in one preprocessed image."""
    temp_dir  = os.path.join(work_root, "temp", stem)
    meta_path = os.path.join(temp_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json missing: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    ground_truth = meta["ground_truth"]
    binary       = cv2.imread(meta["binary_path"], cv2.IMREAD_GRAYSCALE)
    if binary is None:
        raise ValueError(f"Cannot read binary: {meta['binary_path']}")

    with open(meta["segments_path"], "r", encoding="utf-8") as f:
        segments = json.load(f)

    N = len(segments)
    print(f"    {stem}  |  segments: {N}")

    if N == 0:
        results_data = {
            "stem":                     stem,
            "ground_truth":             ground_truth,
            "predicted_text":           "",
            "predicted_text_no_spaces": "",
            "wer":                      100.0,
            "cer":                      100.0,
            "word_spacer_enabled":      cfg.word_spacer_enabled,
            "aksharas":                 [],
        }
    else:
        skeleton     = _build_full_skeleton(binary, cfg)
        char_results = _greedy_segment(skeleton, segments, model,
                                       device, idx_to_class, temp_dir, cfg)

        predicted_text_plain = "".join(r["predicted_char"] for r in char_results)

        if cfg.word_spacer_enabled and meta.get("word_segments_path"):
            ws_path = meta["word_segments_path"]
            if os.path.exists(ws_path):
                with open(ws_path, "r", encoding="utf-8") as f:
                    word_groups = json.load(f)
                predicted_text = _annotate_word_indices(char_results, word_groups)
            else:
                predicted_text = predicted_text_plain
                for ak in char_results:
                    ak["word_index"] = 0
        else:
            predicted_text = predicted_text_plain
            for ak in char_results:
                ak["word_index"] = 0

        wer = compute_wer(predicted_text, ground_truth) if ground_truth else 0.0
        cer = compute_cer(predicted_text, ground_truth) if ground_truth else 0.0

        multi_count = sum(1 for r in char_results if r["window_segs"] > 1)
        n_words     = (max((ak["word_index"] or 0) for ak in char_results) + 1
                       if char_results else 0)

        print(f"           aksharas : {len(char_results)} ({multi_count} multi-seg)")
        if cfg.word_spacer_enabled:
            print(f"           words    : {n_words}")
        try:
            print(f"           predicted: {predicted_text}")
        except UnicodeEncodeError:
            print(f"           predicted: [Sinhala text - encoding error in console]")
        print(f"           WER: {wer}%  CER: {cer}%")

        results_data = {
            "stem":                     stem,
            "ground_truth":             ground_truth,
            "predicted_text":           predicted_text,
            "predicted_text_no_spaces": predicted_text_plain,
            "wer":                      wer,
            "cer":                      cer,
            "n_segments":               N,
            "n_aksharas":               len(char_results),
            "n_multi_seg":              multi_count,
            "n_words":                  n_words if cfg.word_spacer_enabled else None,
            "multi_seg_threshold":      cfg.multi_seg_threshold,
            "word_spacer_enabled":      cfg.word_spacer_enabled,
            "aksharas":                 char_results,
        }

    results_path = os.path.join(temp_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    return results_data

# =============================================================================
# RUN STAGE 3 — CLASSIFICATION
# =============================================================================

def run_stage3_classification(cfg: PipelineConfig,
                               stems: list,
                               work_root: str,
                               model,
                               device: torch.device,
                               idx_to_class: dict) -> list:
    """Classify all preprocessed images sequentially."""
    print(f"  [Stage 3 - Classification] Word spacer : {'ENABLED' if cfg.word_spacer_enabled else 'DISABLED'}")
    print(f"  [Stage 3 - Classification] Classifying {len(stems)} image(s) on {device}...\n")

    all_results = []
    for i, stem in enumerate(stems, 1):
        print(f"    [{i}/{len(stems)}]", end=" ")
        try:
            res = _process_one(stem, work_root, model, device,
                               idx_to_class, cfg)
            all_results.append(res)
            print()
        except Exception as exc:
            import traceback
            print(f"\n           ERROR: {exc}")
            traceback.print_exc()

    # Ensure all pending GPU ops finish before returning
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"  [Stage 3 - Classification] Done.\n")
    return all_results
