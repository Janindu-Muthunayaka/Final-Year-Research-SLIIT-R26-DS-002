"""
Sinhala Character Skeleton Extractor — Batch Mode
===================================================
Reads all PNG images from FontImages folder (recursively through subfolders).

Pipeline:
  1. Polarity detection      → normalise to black text on white background
  2. Pre-process strokes      → closing + dilation to fix dashed/broken lines
  3. Medial axis skeleton     → skimage skeletonize
  4. Output                   → WHITE background, BLACK skeleton lines

OUTPUT STRUCTURE (mirrors input subfolders):
  SkeletonImages/
    vowel_අ_0001/
      0001_Basic_FontName_Skeleton.png

DEBUG OUTPUT (only when SAVE_DEBUG = True, mirrors subfolders):
  SkeletonDebug/
    vowel_අ_0001/
      0001_Basic_FontName_Stg1_Binary.png       ← after polarity + threshold
      0001_Basic_FontName_Stg2_AfterClose.png   ← after morphological closing
      0001_Basic_FontName_Stg3_AfterDilate.png  ← after dilation
      0001_Basic_FontName_Stg4_Skeleton.png     ← final skeleton (same as main output)

Reports/logs → SkeletonReports/

FLAGS:
  FULLSET    = True  → process every image found
  FULLSET    = False → process only 20 randomly sampled fonts (for testing)
  SAVE_DEBUG = True  → also save all 4 pipeline stage images to SkeletonDebug
  SAVE_DEBUG = False → skip debug output (faster, less disk usage)
"""

import cv2
import numpy as np
import os
import random
import logging
import csv
from datetime import datetime
from skimage.morphology import skeletonize

# ─── MASTER SWITCHES ───────────────────────────────────────────────────────────
# True  = full dataset run  |  False = test run (TEST_FONT_SAMPLE fonts only)
FULLSET = True

# True  = save all 4 pipeline stage images to SkeletonDebug folder
# False = skip debug output entirely
SAVE_DEBUG = True

# ─── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR       = r"E:\Sliit\Research\Repositoryv2\Datasets"
INPUT_FOLDER   = os.path.join(BASE_DIR, "FontImages")
OUTPUT_FOLDER  = os.path.join(BASE_DIR, "SkeletonImages")
DEBUG_FOLDER   = os.path.join(BASE_DIR, "SkeletonDebug")
REPORTS_FOLDER = os.path.join(BASE_DIR, "SkeletonReports")

# ─── TEST MODE SETTINGS ────────────────────────────────────────────────────────
TEST_FONT_SAMPLE = 20    # number of unique font stems to keep in test mode
RANDOM_SEED      = 42

# ─── POLARITY DETECTION ────────────────────────────────────────────────────────
POLARITY_SAMPLE_FRACTION = 0.05

# ─── STROKE PRE-PROCESSING (fixes dashed/broken skeletons) ────────────────────
# Morphological closing kernel — fills small internal gaps in thin strokes.
# Increase if gaps still appear; decrease if characters start merging.
# Recommended range: 3–9  (must be odd)
PREPROCESS_CLOSE_K = 5

# Dilation kernel — thickens strokes so skeleton stays connected.
# Recommended range: 3–7  (must be odd)
PREPROCESS_DILATE_K = 3

# Dilation iterations. Increase to 2 if thin fonts still produce gaps.
PREPROCESS_DILATE_ITER = 1

# ─── SKELETON SETTINGS ─────────────────────────────────────────────────────────
# Post-skeleton dilation for output line visibility. 1 = thin, 2 = thicker.
SKELETON_DILATE = 1

# ─── LOGGING ───────────────────────────────────────────────────────────────────
LOG_PATH      = os.path.join(REPORTS_FOLDER, "skeleton_extraction.log")
MANIFEST_PATH = os.path.join(REPORTS_FOLDER, "skeleton_manifest.csv")
ERRORS_PATH   = os.path.join(REPORTS_FOLDER, "skeleton_errors.csv")

# ──────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    os.makedirs(REPORTS_FOLDER, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# POLARITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_polarity(gray, h, w):
    """
    Three-way vote: corners + border strip + histogram.
    Returns True if background is dark (image needs inversion).
    """
    patch      = max(5, int(min(h, w) * POLARITY_SAMPLE_FRACTION))
    votes_dark = 0

    corners = [
        gray[:patch, :patch],
        gray[:patch, w - patch:],
        gray[h - patch:, :patch],
        gray[h - patch:, w - patch:],
    ]
    if np.mean([c.mean() for c in corners]) < 128:
        votes_dark += 1

    border = np.concatenate([
        gray[:patch, :].flatten(),
        gray[-patch:, :].flatten(),
        gray[:, :patch].flatten(),
        gray[:, -patch:].flatten(),
    ])
    if np.mean(border) < 128:
        votes_dark += 1

    hist        = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_smooth = np.convolve(hist, np.ones(15) / 15, mode='same')
    if hist_smooth[:128].sum() > hist_smooth[128:].sum():
        votes_dark += 1

    return votes_dark >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# SKELETON PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def extract_skeleton(src):
    """
    Takes a BGR or grayscale image.

    Returns a dict with all pipeline stages (all are grayscale uint8):
      'stg1_binary'       — after polarity correction + threshold
      'stg2_after_close'  — after morphological closing
      'stg3_after_dilate' — after dilation
      'stg4_skeleton'     — final skeleton (white bg, black lines)

    Raises ValueError if image is blank/invalid.
    """
    # Convert to grayscale if needed
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()

    h, w = gray.shape

    # ── Stage 1: Polarity normalise → black text on white bg ─────────────────
    bg_is_dark = detect_polarity(gray, h, w)
    gray_norm  = cv2.bitwise_not(gray) if bg_is_dark else gray
    # Otsu automatically finds optimal threshold per image — fixes light/thin
    # fonts where a fixed 127 threshold clips strokes into white (dashed lines).
    _, binary  = cv2.threshold(gray_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # letter_mask: 1 where ink exists (ink=0 in binary → mask=1)
    letter_mask = (binary == 0).astype(np.uint8)

    if letter_mask.sum() < 50:
        raise ValueError("Blank or near-blank image — no ink found")

    # Save Stg1 as a visible image: white bg, black ink
    stg1_img = np.full((h, w), 255, dtype=np.uint8)
    stg1_img[letter_mask == 1] = 0

    # ── Stage 2: Morphological closing — fill gaps within strokes ────────────
    k_close     = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (PREPROCESS_CLOSE_K, PREPROCESS_CLOSE_K))
    mask_closed = cv2.morphologyEx(letter_mask, cv2.MORPH_CLOSE, k_close, iterations=1)

    stg2_img = np.full((h, w), 255, dtype=np.uint8)
    stg2_img[mask_closed == 1] = 0

    # ── Stage 3: Dilation — thicken strokes for connected skeleton ───────────
    k_dilate    = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (PREPROCESS_DILATE_K, PREPROCESS_DILATE_K))
    mask_dilated = cv2.dilate(mask_closed, k_dilate, iterations=PREPROCESS_DILATE_ITER)

    stg3_img = np.full((h, w), 255, dtype=np.uint8)
    stg3_img[mask_dilated == 1] = 0

    # ── Stage 4: Medial axis skeleton ────────────────────────────────────────
    skeleton = skeletonize(mask_dilated.astype(bool))
    skel_img = skeleton.astype(np.uint8) * 255

    if SKELETON_DILATE > 0:
        k        = np.ones((3, 3), np.uint8)
        skel_img = cv2.dilate(skel_img, k, iterations=SKELETON_DILATE)

    stg4_img = np.full((h, w), 255, dtype=np.uint8)
    stg4_img[skel_img > 0] = 0

    return {
        "stg1_binary":       stg1_img,
        "stg2_after_close":  stg2_img,
        "stg3_after_dilate": stg3_img,
        "stg4_skeleton":     stg4_img,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE HELPER (Unicode-safe)
# ═══════════════════════════════════════════════════════════════════════════════

def save_png_unicode(path, img):
    """Save a grayscale/BGR image to a Unicode path using imencode + open()."""
    _, buf = cv2.imencode(".png", img)
    with open(path, "wb") as f:
        f.write(buf)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def collect_input_files(input_folder):
    """
    Walk input_folder recursively, return list of absolute PNG paths.
    Skips files that already contain '_Skeleton', '_Stg' in their name
    (avoids re-processing previous outputs).
    """
    all_files = []
    for root, _dirs, files in os.walk(input_folder):
        for fname in files:
            if fname.lower().endswith(".png"):
                if "_Skeleton" not in fname and "_Stg" not in fname:
                    all_files.append(os.path.join(root, fname))
    return all_files


def filter_by_font_sample(files, n, seed):
    """
    Test mode: keep only files whose font_stem matches one of n randomly
    chosen font stems. Filename format: classid_tier_fontstem.png
    """
    stem_map = {}
    unparsed = []

    for fp in files:
        fname = os.path.basename(fp)
        parts = fname.replace(".png", "").split("_", 2)
        if len(parts) == 3:
            stem_map.setdefault(parts[2], []).append(fp)
        else:
            unparsed.append(fp)

    rng          = random.Random(seed)
    chosen_stems = rng.sample(list(stem_map.keys()),
                               min(n, len(stem_map)))
    filtered = []
    for s in chosen_stems:
        filtered.extend(stem_map[s])
    filtered.extend(unparsed[:max(0, n - len(chosen_stems))])
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# CSV HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def open_appender(path, fieldnames):
    write_header = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8-sig")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        w.writeheader()
    return f, w


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    setup_logging()
    os.makedirs(OUTPUT_FOLDER,  exist_ok=True)
    os.makedirs(REPORTS_FOLDER, exist_ok=True)
    if SAVE_DEBUG:
        os.makedirs(DEBUG_FOLDER, exist_ok=True)

    logging.info("=" * 60)
    logging.info("Sinhala Skeleton Extractor — START")
    logging.info(f"Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Mode           : {'FULL DATASET' if FULLSET else f'TEST ({TEST_FONT_SAMPLE} fonts)'}")
    logging.info(f"Debug output   : {'ON  → ' + DEBUG_FOLDER if SAVE_DEBUG else 'OFF'}")
    logging.info(f"Close kernel   : {PREPROCESS_CLOSE_K}px")
    logging.info(f"Dilate kernel  : {PREPROCESS_DILATE_K}px  x{PREPROCESS_DILATE_ITER} iter")
    logging.info(f"Skel dilate    : {SKELETON_DILATE} iter (post)")
    logging.info(f"Input          : {INPUT_FOLDER}")
    logging.info(f"Output         : {OUTPUT_FOLDER}")
    logging.info(f"Reports        : {REPORTS_FOLDER}")
    logging.info("=" * 60)

    # ── Discover ──────────────────────────────────────────────────────────────
    all_files = collect_input_files(INPUT_FOLDER)
    logging.info(f"Total PNG files found : {len(all_files)}")

    if not all_files:
        logging.error("No PNG files found. Exiting.")
        return

    if not FULLSET:
        all_files = filter_by_font_sample(all_files, TEST_FONT_SAMPLE, RANDOM_SEED)
        logging.info(f"Test mode: {len(all_files)} images ({TEST_FONT_SAMPLE} font stems)")

    total = len(all_files)
    logging.info(f"Images to process : {total}")
    logging.info("-" * 60)

    # ── CSVs ──────────────────────────────────────────────────────────────────
    manifest_fields = ["input_path", "output_path", "subfolder",
                       "class_id", "tier", "font_stem"]
    error_fields    = ["input_path", "subfolder", "filename", "reason"]
    mf, mw = open_appender(MANIFEST_PATH, manifest_fields)
    ef, ew = open_appender(ERRORS_PATH,   error_fields)

    success = 0
    errors  = 0

    for idx, input_path in enumerate(all_files, start=1):

        fname     = os.path.basename(input_path)
        rel_dir   = os.path.relpath(os.path.dirname(input_path), INPUT_FOLDER)
        subfolder = rel_dir
        stem      = fname.replace(".png", "")

        # Parse metadata
        parts     = stem.split("_", 2)
        class_id  = parts[0] if len(parts) > 0 else ""
        tier      = parts[1] if len(parts) > 1 else ""
        font_stem = parts[2] if len(parts) > 2 else ""

        # Output paths
        out_dir  = os.path.join(OUTPUT_FOLDER, rel_dir)
        out_path = os.path.join(out_dir, f"{stem}_Skeleton.png")
        os.makedirs(out_dir, exist_ok=True)

        # Debug paths
        if SAVE_DEBUG:
            dbg_dir = os.path.join(DEBUG_FOLDER, rel_dir)
            os.makedirs(dbg_dir, exist_ok=True)
            dbg_paths = {
                "stg1_binary":       os.path.join(dbg_dir, f"{stem}_Stg1_Binary.png"),
                "stg2_after_close":  os.path.join(dbg_dir, f"{stem}_Stg2_AfterClose.png"),
                "stg3_after_dilate": os.path.join(dbg_dir, f"{stem}_Stg3_AfterDilate.png"),
                "stg4_skeleton":     os.path.join(dbg_dir, f"{stem}_Stg4_Skeleton.png"),
            }

        print(
            f"\r[{idx}/{total}]  ✓{success}  ✗{errors}  "
            f"{subfolder[:28]:<28}  {fname[:36]}   ",
            end="", flush=True
        )

        # ── Load (Unicode-safe) ───────────────────────────────────────────────
        try:
            raw = np.fromfile(input_path, dtype=np.uint8)
            src = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception:
            src = None

        if src is None:
            msg = "Cannot open file (imdecode failed)"
            logging.warning(f"  [SKIP] {fname} — {msg}")
            ew.writerow({"input_path": input_path, "subfolder": subfolder,
                         "filename": fname, "reason": msg})
            errors += 1
            continue

        # ── Process ───────────────────────────────────────────────────────────
        try:
            stages = extract_skeleton(src)
        except Exception as ex:
            msg = str(ex)
            logging.warning(f"  [ERROR] {fname} — {msg}")
            ew.writerow({"input_path": input_path, "subfolder": subfolder,
                         "filename": fname, "reason": msg})
            errors += 1
            continue

        # ── Save main skeleton output ─────────────────────────────────────────
        try:
            save_png_unicode(out_path, stages["stg4_skeleton"])
        except Exception as ex:
            msg = f"Save failed: {ex}"
            logging.warning(f"  [ERROR] {fname} — {msg}")
            ew.writerow({"input_path": input_path, "subfolder": subfolder,
                         "filename": fname, "reason": msg})
            errors += 1
            continue

        # ── Save debug stage images ───────────────────────────────────────────
        if SAVE_DEBUG:
            for stage_key, dbg_path in dbg_paths.items():
                try:
                    save_png_unicode(dbg_path, stages[stage_key])
                except Exception as ex:
                    logging.warning(f"  [DEBUG SAVE FAIL] {stage_key}: {ex}")

        # ── Manifest ──────────────────────────────────────────────────────────
        mw.writerow({
            "input_path":  input_path,
            "output_path": out_path,
            "subfolder":   subfolder,
            "class_id":    class_id,
            "tier":        tier,
            "font_stem":   font_stem,
        })
        success += 1

        if success % 200 == 0:
            mf.flush(); ef.flush()

    print()
    mf.flush(); mf.close()
    ef.flush(); ef.close()

    logging.info("=" * 60)
    logging.info("EXTRACTION COMPLETE")
    logging.info(f"  Successfully processed : {success}")
    logging.info(f"  Errors / skipped       : {errors}")
    logging.info(f"  Total attempted        : {total}")
    logging.info(f"  Manifest : {MANIFEST_PATH}")
    logging.info(f"  Errors   : {ERRORS_PATH}")
    logging.info(f"  Log      : {LOG_PATH}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()