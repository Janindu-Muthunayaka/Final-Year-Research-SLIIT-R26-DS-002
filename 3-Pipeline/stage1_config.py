# =============================================================================
# stage1_config.py  —  Configuration, Constants & PipelineConfig
#
# All tunable parameters, path constants, GPU configuration, and the
# PipelineConfig dataclass live here.  Every other stage module imports
# from this file — never the other way around.
# =============================================================================

from __future__ import annotations

import os
import math
from dataclasses import dataclass

import torch

# =============================================================================
# ██████████████████████  PATH CONSTANTS  █████████████████████████████████████
# =============================================================================

INPUT_FOLDER = r"E:\Sliit\Research\Repositoryv2\Datasets\TestData\Full30k\Images"
LABEL_CSV    = r"E:\Sliit\Research\Repositoryv2\Datasets\TestData\Full30k\Label_List.csv"
TESSERACT_CSV = r"E:\Sliit\Research\Repositoryv2\Datasets\TestData\Tessaract_Result_TextCleaned.csv"
MODEL_PATH   = r"E:\Sliit\Research\Repositoryv2\Final-Year-Research-SLIIT-R26-DS-002\2-Model\final_model.pth"
CLASS_MAP    = r"E:\Sliit\Research\Repositoryv2\Final-Year-Research-SLIIT-R26-DS-002\2-Model\class_mapping.json"
VARIANTS_PATH = r"E:\Sliit\Research\Repositoryv2\Final-Year-Research-SLIIT-R26-DS-002\2-Model\Variants.py"
WORK_ROOT    = r"E:\Sliit\Research\Repositoryv2\Final-Year-Research-SLIIT-R26-DS-002\3-Pipeline"

# =============================================================================
# ██████████████████████  SAMPLING DEFAULTS  ██████████████████████████████████
# =============================================================================

DEFAULT_FULLSET = False
DEFAULT_SAMPLE  = 50
DEFAULT_SEED    = 42

# =============================================================================
# ██████████████████████  STAGE-2 PREPROCESSING PARAMETERS (TUNABLE)  █████████
# =============================================================================

P_TARGET_HEIGHT    = 512
P_SMOOTHING_K      = 3
P_CLOSE_K          = 3
P_SKELETON_DIL     = 1
P_VALLEY_MIN_WIDTH = 2

# =============================================================================
# ██████████████████████  STAGE-3 SEGMENTATION PARAMETERS (TUNABLE)  █████████
# =============================================================================

# Minimum pixel area for a connected component to be treated as a valid blob.
# Components smaller than this are discarded as noise.
# Tune upward if stray ink specks are being counted as characters.
P_BLOB_MIN_AREA = 15      # Reduced from 30 to capture smaller components
P_RECT_THRESHOLD  = 1.3    # Width/Height ratio to trigger thin-valley splitting
P_THIN_RATIO      = 0.05   # Ink height ratio (ink/total_h) to treat as a 'thin' valley
P_MIN_SPLIT_DIST  = 12     # Min horizontal pixels between character splits

# =============================================================================
# ██████████████████████  STAGE-4 CLASSIFIER PARAMETERS (TUNABLE)  ████████████
# =============================================================================

P_CHAR_CANVAS_SIZE    = 384   # ← do NOT change unless model is retrained
P_WINDOW_PAD          = 12
P_MULTI_SEG_THRESHOLD = 97.0

# =============================================================================
# ██████████████████████  WORD-SPACER PARAMETERS (TUNABLE)  ███████████████████
# =============================================================================

P_WORD_SPACER_ENABLED = True
P_WORD_GAP_PX         = 50

# =============================================================================
# ██████████████████████  STAGE-4 MISC  ███████████████████████████████████████
# =============================================================================

TOP_K = 5   # alternative predictions to retain — not tunable

# =============================================================================
# ██████████████████████  ImageNet NORMALISATION  █████████████████████████████
# =============================================================================

_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_NORM_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

# =============================================================================
# ██████████████████████  GPU / HARDWARE CONFIG  ██████████████████████████████
# =============================================================================

# Maximum crops sent to GPU in one forward pass.
INFER_BATCH_SIZE: int = 64

# Stage-2 parallel workers.
S1_WORKERS: int = max(1, math.ceil((os.cpu_count() or 8) * 0.75))

# PNG write pool size (I/O bound, kept small).
PNG_POOL_WORKERS: int = 4

# Enable CUDA benchmark mode for fixed-size inference loops.
CUDNN_BENCHMARK: bool = True

# =============================================================================
# ██████████████████████  DEVICE SELECTION  ███████████████████████████████████
# =============================================================================

def _select_device() -> torch.device:
    """
    Returns cuda:0 if available, else cpu.
    Applies cudnn.benchmark and prints a one-line device summary.
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = False
        props = torch.cuda.get_device_properties(dev)
        vram_gb = props.total_memory / 1024 ** 3
        print(f"  [Device] CUDA  - {props.name}  "
              f"VRAM {vram_gb:.1f} GB  "
              f"cudnn.benchmark={CUDNN_BENCHMARK}")
        if vram_gb < 4.0:
            print(f"  [Device] WARNING: < 4 GB VRAM detected. "
                  f"Consider reducing INFER_BATCH_SIZE (currently {INFER_BATCH_SIZE}).")
    else:
        dev = torch.device("cpu")
        print(f"  [Device] CPU   - CUDA not available")
    return dev


# Module-level device resolved once at import time.
_DEVICE: torch.device = _select_device()

# =============================================================================
# ██████████████████████  PIPELINE CONFIG DATACLASS  ██████████████████████████
# =============================================================================

@dataclass
class PipelineConfig:
    """Single object carrying every tunable parameter through the pipeline."""
    # Paths
    input_folder:        str   = INPUT_FOLDER
    label_csv:           str   = LABEL_CSV
    model_path:          str   = MODEL_PATH
    class_map:           str   = CLASS_MAP
    work_root:           str   = WORK_ROOT

    # Sampling
    fullset:             bool  = DEFAULT_FULLSET
    sample:              int   = DEFAULT_SAMPLE
    seed:                int   = DEFAULT_SEED

    # Stage-2 preprocessing
    target_height:       int   = P_TARGET_HEIGHT
    smoothing_k:         int   = P_SMOOTHING_K
    close_k:             int   = P_CLOSE_K
    skeleton_dil:        int   = P_SKELETON_DIL
    valley_min_width:    int   = P_VALLEY_MIN_WIDTH

    # Stage-3 segmentation
    blob_min_area:       int   = P_BLOB_MIN_AREA
    rect_threshold:      float = P_RECT_THRESHOLD
    thin_ratio:          float = P_THIN_RATIO
    min_split_dist:      int   = P_MIN_SPLIT_DIST

    # Stage-4 classifier
    char_canvas_size:    int   = P_CHAR_CANVAS_SIZE
    window_pad:          int   = P_WINDOW_PAD
    multi_seg_threshold: float = P_MULTI_SEG_THRESHOLD
    variants_path:       str   = VARIANTS_PATH

    # Word spacer
    word_spacer_enabled: bool  = P_WORD_SPACER_ENABLED
    word_gap_px:         int   = P_WORD_GAP_PX

    def as_param_dict(self) -> dict:
        return {
            "target_height":       self.target_height,
            "smoothing_k":         self.smoothing_k,
            "close_k":             self.close_k,
            "skeleton_dil":        self.skeleton_dil,
            "valley_min_width":    self.valley_min_width,
            "blob_min_area":       self.blob_min_area,
            "rect_threshold":      self.rect_threshold,
            "thin_ratio":          self.thin_ratio,
            "min_split_dist":      self.min_split_dist,
            "window_pad":          self.window_pad,
            "multi_seg_threshold": self.multi_seg_threshold,
            "word_spacer_enabled": self.word_spacer_enabled,
            "word_gap_px":         self.word_gap_px,
        }