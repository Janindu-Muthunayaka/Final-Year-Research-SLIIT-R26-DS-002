# =============================================================================
# stage2_preprocessing.py  —  Image Preprocessing & Skeletonisation
#
# Handles the entire Stage-2 preprocessing pipeline:
#   raw image → greyscale → binary → rough skeleton → refined skeleton
#
# INPUT:
#   - all_files   : list[str]       — absolute paths to input images
#   - labels      : dict[str, str]  — {stem: ground_truth_text}
#   - cfg         : PipelineConfig  — tunable parameters
#   - temp_root   : str             — directory for intermediate artefacts
#
# OUTPUT (from run_stage2_preprocessing):
#   - list[dict]  — per-image metadata dicts
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
    labels = {}
    if not os.path.exists(path):
        print(f"  [WARN] Label file not found: {path}")
        return labels
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            idx = line.find(",")
            if idx == -1: continue
            labels[line[:idx].strip()] = line[idx+1:].strip().strip('"')
    return labels

def _save_png(numpy_img: np.ndarray, path: str) -> None:
    _, buf = cv2.imencode(".png", numpy_img)
    with open(path, "wb") as fh:
        fh.write(buf)

# =============================================================================
# PREPROCESSING
# =============================================================================

def _preprocess_sentence(src_img: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    gray = (cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
            if len(src_img.shape) == 3 else src_img.copy())
    h, w = gray.shape
    scale = cfg.target_height / h
    new_size = (max(1, int(w * scale)), cfg.target_height)
    resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)
    k = cfg.smoothing_k if cfg.smoothing_k % 2 == 1 else cfg.smoothing_k + 1
    blurred = cv2.GaussianBlur(resized, (k, k), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    return binary

# =============================================================================
# SKELETONISATION
# =============================================================================

def _skeletonize_roi(roi: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Skeletonize a single ROI (used for rough skeleton)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_k, cfg.close_k))
    healed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k)
    skel = sk_skeletonize(healed > 0).astype(np.uint8) * 255
    if cfg.skeleton_dil > 0:
        skel = cv2.dilate(skel, np.ones((3, 3), np.uint8), iterations=cfg.skeleton_dil)
    return skel

def _build_sentence_skeleton(binary: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Build a rough-sentence skeleton by skeletonizing each contour ROI."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(binary)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch < 30: continue
        canvas[y:y+ch, x:x+cw] = cv2.bitwise_or(canvas[y:y+ch, x:x+cw],
                                                _skeletonize_roi(binary[y:y+ch, x:x+cw], cfg))
    return canvas

def _build_full_skeleton(binary: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Build a refined full-image skeleton using global morphology (optimal for classification)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_k, cfg.close_k))
    healed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    skel = sk_skeletonize(healed > 0).astype(np.uint8) * 255
    if cfg.skeleton_dil > 0:
        skel = cv2.dilate(skel, np.ones((3, 3), np.uint8), iterations=cfg.skeleton_dil)
    return skel

# =============================================================================
# PROCESS ONE IMAGE
# =============================================================================

def _process_one(img_path: str,
                 ground_truth: str,
                 temp_root: str,
                 cfg: PipelineConfig) -> dict:
    fname = os.path.basename(img_path)
    stem  = os.path.splitext(fname)[0]
    out   = os.path.join(temp_root, stem)
    os.makedirs(out, exist_ok=True)

    raw = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None: raise ValueError(f"Could not decode image: {img_path}")

    orig_out = os.path.join(out, fname)
    shutil.copy2(img_path, orig_out)

    binary = _preprocess_sentence(img, cfg)
    
    # 1. Build rough skeleton (contour-based)
    skel_rough = _build_sentence_skeleton(binary, cfg)
    
    # 2. Build refined skeleton (global morphological - classification optimized)
    skel_refined = _build_full_skeleton(binary, cfg)

    binary_path = os.path.join(out, "binary.png")
    skel_path = os.path.join(out, "skeleton.png")
    skel_refined_path = os.path.join(out, "skeleton_refined.png")
    
    _save_png(binary, binary_path)
    _save_png(cv2.bitwise_not(skel_rough), skel_path)
    _save_png(cv2.bitwise_not(skel_refined), skel_refined_path)

    meta = {
        "stem": stem,
        "fname": fname,
        "ground_truth": ground_truth,
        "orig_path": orig_out,
        "binary_path": binary_path,
        "skel_path": skel_path,
        "skel_refined_path": skel_refined_path,
        "temp_dir": out,
        "params": {
            "TARGET_HEIGHT": cfg.target_height,
            "SMOOTHING_K": cfg.smoothing_k,
            "CLOSE_K": cfg.close_k,
            "SKELETON_DIL": cfg.skeleton_dil,
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
    print(f"  [Stage 2 - Preprocessing] Processing {len(all_files)} image(s)...")

    def _worker(img_path: str) -> dict:
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        ground_truth = labels.get(stem, "")
        return _process_one(img_path, ground_truth, temp_root, cfg)

    metas = []
    n = len(all_files)
    completed = 0
    workers = min(S1_WORKERS, n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_path = {pool.submit(_worker, p): p for p in all_files}
        for future in concurrent.futures.as_completed(future_to_path):
            fname = os.path.basename(future_to_path[future])
            completed += 1
            try:
                meta = future.result()
                print(f"    [{completed}/{n}] {fname}  (processed)")
                metas.append(meta)
            except Exception as exc:
                print(f"    [{completed}/{n}] {fname}  ERROR: {exc}")

    # Deterministic ordering
    path_order = {p: i for i, p in enumerate(all_files)}
    metas.sort(key=lambda m: path_order.get(next((p for p in all_files if os.path.splitext(os.path.basename(p))[0] == m["stem"]), ""), 0))

    image_list_path = os.path.join(temp_root, "image_list.json")
    with open(image_list_path, "w", encoding="utf-8") as f:
        json.dump([m["stem"] for m in metas], f, ensure_ascii=False, indent=2)

    print(f"  [Stage 2 - Preprocessing] Done - temp artefacts in: {temp_root}\n")
    return metas
