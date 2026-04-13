import os
import json
import cv2
import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize

class CFG:
    target_height = 120
    smoothing_k = 3
    close_k = 5
    skeleton_dil = 1
    valley_min_width = 5
    word_gap_px = 30
    word_spacer_enabled = True

cfg = CFG()

def _skeletonize_roi(roi, cfg):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_k, cfg.close_k))
    healed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k)
    skel = sk_skeletonize(healed > 0).astype(np.uint8) * 255
    if cfg.skeleton_dil > 0:
        skel = cv2.dilate(skel, np.ones((3, 3), np.uint8), iterations=cfg.skeleton_dil)
    return skel

def _build_sentence_skeleton(binary, cfg):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(binary)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch < 30: continue
        canvas[y:y+ch, x:x+cw] = cv2.bitwise_or(canvas[y:y+ch, x:x+cw],
                                                _skeletonize_roi(binary[y:y+ch, x:x+cw], cfg))
    return canvas

def _build_full_skeleton(binary, cfg):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_k, cfg.close_k))
    healed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    skel = sk_skeletonize(healed > 0).astype(np.uint8) * 255
    if cfg.skeleton_dil > 0:
        skel = cv2.dilate(skel, np.ones((3, 3), np.uint8), iterations=cfg.skeleton_dil)
    return skel

def _find_valley_segments(skeleton_canvas, cfg):
    projection = (skeleton_canvas > 0).sum(axis=0).astype(int)
    W = projection.shape[0]
    in_seg = False
    seg_start = 0
    segments = []
    for x in range(W):
        if projection[x] > 0 and not in_seg:
            seg_start = x
            in_seg = True
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

# Test on a real image from the temp dir if possible, otherwise skip
temp_dir = r"e:\Sliit\Research\Repositoryv2\Final-Year-Research-SLIIT-R26-DS-002\3-Pipeline\temp"
if os.path.exists(temp_dir):
    stems = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
    if stems:
        stem = stems[0]
        meta_path = os.path.join(temp_dir, stem, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f: meta = json.load(f)
        binary = cv2.imread(meta["binary_path"], cv2.IMREAD_GRAYSCALE)
        
        skel_rough = _build_sentence_skeleton(binary, cfg)
        skel_refined = _build_full_skeleton(binary, cfg)
        
        segs_rough = _find_valley_segments(skel_rough, cfg)
        segs_refined = _find_valley_segments(skel_refined, cfg)
        
        print(f"Stem: {stem}")
        print(f"Rough segments: {len(segs_rough)}")
        print(f"Refined segments: {len(segs_refined)}")
        if segs_rough != segs_refined:
            print("DIFFERENCE DETECTED in segment boundaries!")
            print(f"Rough: {segs_rough[:5]}")
            print(f"Refined: {segs_refined[:5]}")
        else:
            print("Segments are identical.")
