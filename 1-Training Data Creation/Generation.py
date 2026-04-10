"""
Sinhala Character Dataset Generator  v4
=========================================
Generates 512×512 PNG images for each character class × selected fonts.

TWO RENDERING PATHS:
  Unicode fonts  — sends Unicode string directly to Qt renderer
  Legacy fonts   — converts via FM Abhaya mapping, sends ASCII string

OUTPUT STRUCTURE (one folder per class, class_id at end of name):
  FontImages/
    vowel_අ_0001/
      0001_Basic_FontName.png
      0001_Hard_FontName.png
    hal_ක්_0017/
      0017_Basic_FontName.png
    combo_කා_0019/
      0019_Intermediate_FontName.png

Folder name : {category_short}_{rendered_char}_{class_id}
File name   : {class_id}_{tier}_{font_stem}.png

Category short: vowel / hal / combo
Class_id at END of folder name — Explorer sorts by type+char first.
Font stem at END of file name — inconsistent names don't break sorting.

OUTPUTS (all written to script folder):
  font_selection.csv   — fonts picked, tier, split, font_type
  manifest.csv         — every generated image + full metadata
  errors.csv           — every skipped render + reason
  generation.log       — full timestamped run log

SAFETY:
  - Resumes where it left off (scans existing PNGs)
  - Auto-shutdown: 100 consecutive errors OR >40% error rate
  - fm_mapping failures NOT counted as hard errors

REQUIREMENTS:
  pip install PyQt5 numpy fonttools
"""

import sys
import csv
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFontDatabase, QFont, QPainter, QImage, QColor, QFontMetrics
from PyQt5.QtCore import Qt

try:
    from fontTools import ttLib as _ttLib
    import logging as _ftlog
    _ftlog.getLogger("fontTools").setLevel(_ftlog.CRITICAL)
    FONTTOOLS_OK = True
except ImportError:
    FONTTOOLS_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent.resolve()

FONT_DIRS = {
    "Basic":        Path(r"E:\Sliit\Research\Repositoryv2\Datasets\Fontpacks\1.Basic"),
    "Intermediate": Path(r"E:\Sliit\Research\Repositoryv2\Datasets\Fontpacks\2.Intermediate"),
    "Hard":         Path(r"E:\Sliit\Research\Repositoryv2\Datasets\Fontpacks\3.Hard"),
}

OUTPUT_DIR      = Path(r"E:\Sliit\Research\Repositoryv2\Datasets\FontImages")
CLASS_LIST_PATH = SCRIPT_DIR / "class_list.csv"

FONT_SELECTION_PATH = SCRIPT_DIR / "font_selection.csv"
MANIFEST_PATH       = SCRIPT_DIR / "manifest.csv"
ERRORS_PATH         = SCRIPT_DIR / "errors.csv"
LOG_PATH            = SCRIPT_DIR / "generation.log"

# ── Sampling ──────────────────────────────────────────────────────────────────
# After manual deduplication: Basic≈80, Intermediate≈100, Hard≈140
# None = take every font found in the folder
SAMPLE_COUNTS = {
    "Basic":        None,
    "Intermediate": None,
    "Hard":         None,
}
TRAIN_RATIO = 0.70   # 70% train, 30% test per tier
RANDOM_SEED = 42

# ── Render ────────────────────────────────────────────────────────────────────
IMAGE_SIZE  = 512
RENDER_PT   = 320
CANVAS_SIZE = 1024
PADDING_PX  = 20

# ── Safety ────────────────────────────────────────────────────────────────────
MAX_CONSECUTIVE_ERRORS = 100
MAX_ERROR_RATE         = 0.40
MIN_RENDERS_FOR_RATE   = 200
FLUSH_EVERY            = 100

# ══════════════════════════════════════════════════════════════════════════════
#  FM ABHAYA MAPPING
# ══════════════════════════════════════════════════════════════════════════════

SINGLE: dict[int, str] = {
    0x0D9A: 'l',   # ක      0x0D9B: 'L',   # ඛ — added below
    0x0D9C: '.',   # ග
    0x0D9D: '>',   # ඝ
    0x0D9E: 'X',   # ඞ
    0x0DA0: 'p',   # ච
    0x0DA1: 'P',   # ඡ
    0x0DA2: 'c',   # ජ
    0x0DA4: '[',   # ඤ
    0x0DA5: '{',   # ඥ
    0x0DA7: 'g',   # ට
    0x0DA8: 'G',   # ඨ
    0x0DA9: 'v',   # ඩ
    0x0DAA: 'V',   # ඪ
    0x0DAB: 'K',   # ණ
    0x0DAC: '~',   # ඬ
    0x0DAD: ';',   # ත
    0x0DAE: ':',   # ථ
    0x0DAF: 'o',   # ද
    0x0DB0: 'O',   # ධ
    0x0DB1: 'k',   # න
    0x0DB3: '|',   # ඳ
    0x0DB4: 'm',   # ප
    0x0DB5: 'M',   # ඵ
    0x0DB6: 'n',   # බ
    0x0DB7: 'N',   # භ
    0x0DB8: 'u',   # ම
    0x0DB9: 'U',   # ඹ
    0x0DBA: 'h',   # ය
    0x0DBB: 'r',   # ර
    0x0DBD: ',',   # ල
    0x0DC0: 'j',   # ව
    0x0DC1: 'Y',   # ශ
    0x0DC2: 'I',   # ෂ
    0x0DC3: 'i',   # ස
    0x0DC4: 'y',   # හ
    0x0DC5: '<',   # ළ
    0x0DC6: '*',   # ෆ
    0x0D85: 'w',   # අ
    0x0D89: 'b',   # ඉ
    0x0D8A: 'B',   # ඊ
    0x0D8B: 'W',   # උ
    0x0D8D: 'R',   # ඍ
    0x0D91: 't',   # එ
    0x0D94: 'T',   # ඔ
    0x0DCA: 'a',   # ්
    0x0DCF: 'd',   # ා
    0x0DD0: 'e',   # ැ
    0x0DD1: 'E',   # ෑ
    0x0DD2: 's',   # ි
    0x0DD3: 'S',   # ී
    0x0DD4: 'q',   # ු
    0x0DD6: 'Q',   # ූ
    0x0DD8: 'D',   # ෘ
    0x0D82: 'x',   # ං
    0x0D9B: 'L',   # ඛ
}

MULTI: dict[int, str] = {
    0x0D86: 'wd',
    0x0D87: 'we',
    0x0D88: 'wE',
    0x0D92: 'ta',
    0x0D93: 'ft',
    0x0D95: 'Ta',
    0x0D96: 'T!',
    0x0D8C: 'W!',
    0x0D8E: 'RD',
    0x0DF2: 'DD',
    0x0D9F: '`.',
    0x0DA3: 'CO',
    0x0DA6: '`P',
}

F_VOWELS: dict[int, tuple[str, str]] = {
    0x0DD9: ('f',  ''),
    0x0DDA: ('f',  'a'),
    0x0DDC: ('f',  'd'),
    0x0DDD: ('f',  'da'),
    0x0DDE: ('f',  '!'),
    0x0DDB: ('ff', ''),
}


def build_fm_string(rendered: str) -> str | None:
    cps = [ord(c) for c in rendered]
    if len(cps) == 1:
        cp = cps[0]
        if cp in SINGLE: return SINGLE[cp]
        if cp in MULTI:  return MULTI[cp]
        return None
    if len(cps) == 2:
        base_cp, sign_cp = cps[0], cps[1]
        base_key = SINGLE.get(base_cp) or MULTI.get(base_cp)
        if base_key is None: return None
        if sign_cp == 0x0DCA: return base_key + 'a'
        if sign_cp == 0x0D82: return base_key + 'x'
        if sign_cp == 0x0DF2: return base_key + 'DD'
        if sign_cp in F_VOWELS:
            pre, suf = F_VOWELS[sign_cp]
            return pre + base_key + suf
        if sign_cp in SINGLE: return base_key + SINGLE[sign_cp]
        return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  FONT TYPE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_font_type(font_path: str) -> str:
    if not FONTTOOLS_OK:
        return 'unicode'
    try:
        tt = _ttLib.TTFont(font_path, lazy=True)
        cmap_table = tt.get("cmap")
        if cmap_table:
            for table in cmap_table.tables:
                for cp in table.cmap:
                    if 0x0D80 <= cp <= 0x0DFF:
                        tt.close()
                        return 'unicode'
        tt.close()
        return 'legacy'
    except Exception:
        return 'legacy'


def detect_all_fonts(selection: list[dict]) -> list[dict]:
    logging.info("Detecting font types…")
    u = l = 0
    for entry in selection:
        ft = detect_font_type(entry["font_path"])
        entry["font_type"] = ft
        if ft == 'unicode': u += 1
        else:               l += 1
    logging.info(f"  Unicode : {u}   Legacy : {l}")
    return selection


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS LIST
# ══════════════════════════════════════════════════════════════════════════════

def load_class_list(path: Path) -> list[dict]:
    classes = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            classes.append(row)
    logging.info(f"Loaded {len(classes)} classes from {path.name}")
    return classes


# ══════════════════════════════════════════════════════════════════════════════
#  FONT DISCOVERY & SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def discover_fonts(font_dirs: dict) -> dict:
    found = {}
    for tier, folder in font_dirs.items():
        if not folder.exists():
            logging.warning(f"Folder not found: {folder}")
            found[tier] = []
            continue
        fonts = sorted(f for f in folder.iterdir()
                       if f.suffix.lower() in (".ttf", ".otf"))
        logging.info(f"  {tier}: {len(fonts)} fonts found")
        found[tier] = fonts
    return found


def sample_fonts(found: dict, sample_counts: dict, seed: int) -> list[dict]:
    rng       = random.Random(seed)
    selection = []
    for tier, fonts in found.items():
        n      = sample_counts.get(tier)
        chosen = list(fonts) if (n is None or n >= len(fonts)) \
                 else rng.sample(fonts, n)
        rng.shuffle(chosen)
        n_train = max(1, int(len(chosen) * TRAIN_RATIO))
        for i, fp in enumerate(chosen):
            selection.append({
                "font_path": str(fp),
                "font_stem": fp.stem,
                "tier":      tier,
                "split":     "train" if i < n_train else "test",
                "font_type": "",
            })
        logging.info(f"  {tier}: {len(chosen)} fonts  "
                     f"({n_train} train / {len(chosen) - n_train} test)")
    return selection


def save_font_selection(selection: list[dict], path: Path):
    fields = ["font_path", "font_stem", "tier", "split", "font_type"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(selection)
    logging.info(f"Font selection saved → {path.name}  ({len(selection)} fonts)")


def load_font_selection(path: Path) -> list[dict]:
    sel = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sel.append(row)
    logging.info(f"Loaded existing font selection ({len(sel)} fonts)")
    return sel


# ══════════════════════════════════════════════════════════════════════════════
#  FOLDER MAP
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_PREFIX = {
    "independent_vowel": "vowel",
    "hal":               "hal",
    "combination":       "combo",
}


def class_folder_name(cls: dict) -> str:
    """
    {category_short}_{rendered_char}_{class_id}
    Examples:  vowel_අ_0001   hal_ක්_0017   combo_කා_0019
    Class_id at end so Explorer sorts by type then character.
    """
    prefix = CATEGORY_PREFIX.get(cls["category"], cls["category"])
    return f"{prefix}_{cls['rendered']}_{cls['class_id']}"


def create_class_folders(classes: list[dict], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    folder_map = {}
    for cls in classes:
        folder = output_dir / class_folder_name(cls)
        folder.mkdir(exist_ok=True)
        folder_map[cls["class_id"]] = folder
    logging.info(f"754 class folders ready under {output_dir}")
    return folder_map


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME
# ══════════════════════════════════════════════════════════════════════════════

def build_done_set(folder_map: dict) -> set[tuple[str, str]]:
    """
    Scan class folders for existing PNGs.
    File name format: {class_id}_{tier}_{font_stem}.png
    Returns set of (class_id, font_stem) already rendered.
    """
    done = set()
    for class_id, folder in folder_map.items():
        for img in folder.glob("*.png"):
            parts = img.stem.split("_", 2)
            if len(parts) == 3:
                done.add((class_id, parts[2]))
    logging.info(f"Resume scan: {len(done)} images already exist")
    return done


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render_and_crop(text: str, font_family: str) -> QImage | None:
    canvas = QImage(CANVAS_SIZE, CANVAS_SIZE, QImage.Format_Grayscale8)
    canvas.fill(255)

    font = QFont(font_family, RENDER_PT)
    fm   = QFontMetrics(font)
    br   = fm.boundingRect(text)

    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.TextAntialiasing, False)
    painter.setRenderHint(QPainter.Antialiasing,     False)
    painter.setFont(font)
    painter.setPen(QColor(0, 0, 0))
    draw_x = (CANVAS_SIZE - br.width())  // 2 - br.x()
    draw_y = (CANVAS_SIZE - br.height()) // 2 - br.y()
    painter.drawText(draw_x, draw_y, text)
    painter.end()

    ptr = canvas.bits()
    ptr.setsize(CANVAS_SIZE * CANVAS_SIZE)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(CANVAS_SIZE, CANVAS_SIZE).copy()

    if np.sum(arr < 250) < 10:
        return None

    row_mask = np.any(arr < 250, axis=1)
    col_mask = np.any(arr < 250, axis=0)
    rmin, rmax = np.where(row_mask)[0][[0, -1]]
    cmin, cmax = np.where(col_mask)[0][[0, -1]]
    cropped = arr[rmin:rmax + 1, cmin:cmax + 1]

    ch, cw  = cropped.shape
    target  = IMAGE_SIZE - 2 * PADDING_PX
    if ch > target or cw > target:
        scale   = target / max(ch, cw)
        new_h   = max(1, int(ch * scale))
        new_w   = max(1, int(cw * scale))
        row_idx = (np.arange(new_h) * ch / new_h).astype(int)
        col_idx = (np.arange(new_w) * cw / new_w).astype(int)
        cropped = cropped[np.ix_(row_idx, col_idx)]
        ch, cw  = new_h, new_w

    out  = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8)
    top  = (IMAGE_SIZE - ch) // 2
    left = (IMAGE_SIZE - cw) // 2
    out[top:top + ch, left:left + cw] = cropped

    result = QImage(out.tobytes(), IMAGE_SIZE, IMAGE_SIZE,
                    IMAGE_SIZE, QImage.Format_Grayscale8)
    return result.copy()


# ══════════════════════════════════════════════════════════════════════════════
#  CSV HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def open_csv_appender(path: Path, fieldnames: list, write_header: bool):
    f = open(path, "a", newline="", encoding="utf-8-sig")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        w.writeheader()
    return f, w


def log_error(writer, class_id, rendered, font_stem,
              tier, split, font_type, reason):
    writer.writerow({
        "class_id": class_id, "rendered":   rendered,
        "font_stem": font_stem, "tier":     tier,
        "split":    split,     "font_type": font_type,
        "reason":   reason,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  SHUTDOWN CHECKER
# ══════════════════════════════════════════════════════════════════════════════

def check_shutdown(consecutive: int, total_errors: int,
                   total_attempted: int) -> bool:
    if consecutive >= MAX_CONSECUTIVE_ERRORS:
        print()
        logging.critical(
            f"SHUTDOWN: {consecutive} consecutive errors. Aborting.")
        return True
    if total_attempted >= MIN_RENDERS_FOR_RATE:
        rate = total_errors / total_attempted
        if rate >= MAX_ERROR_RATE:
            print()
            logging.critical(
                f"SHUTDOWN: error rate {rate:.1%} > {MAX_ERROR_RATE:.0%} "
                f"({total_errors}/{total_attempted}). Aborting.")
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    setup_logging()
    logging.info("=" * 60)
    logging.info("Sinhala Dataset Generator v4 — START")
    logging.info(f"Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Image size : {IMAGE_SIZE}×{IMAGE_SIZE}px")
    logging.info(f"Train ratio: {TRAIN_RATIO:.0%} per tier")
    logging.info(f"fonttools  : {'OK' if FONTTOOLS_OK else 'NOT FOUND'}")
    logging.info("=" * 60)

    classes = load_class_list(CLASS_LIST_PATH)

    # Font selection
    if FONT_SELECTION_PATH.exists():
        logging.info("Existing font_selection.csv — resume mode")
        selection = load_font_selection(FONT_SELECTION_PATH)
        if any(e.get("font_type", "") == "" for e in selection):
            logging.info("Back-filling font_type…")
            selection = detect_all_fonts(selection)
            save_font_selection(selection, FONT_SELECTION_PATH)
    else:
        logging.info("Fresh run — scanning fonts")
        found     = discover_fonts(FONT_DIRS)
        selection = sample_fonts(found, SAMPLE_COUNTS, RANDOM_SEED)
        selection = detect_all_fonts(selection)
        save_font_selection(selection, FONT_SELECTION_PATH)

    # Class folders
    folder_map = create_class_folders(classes, OUTPUT_DIR)

    # Resume
    done_set = build_done_set(folder_map)

    # Work list
    work = [
        (fi, cls)
        for fi in selection
        for cls in classes
        if (cls["class_id"], fi["font_stem"]) not in done_set
    ]

    total_work   = len(work)
    already_done = len(done_set)

    logging.info(f"Total renders needed   : {len(selection) * len(classes)}")
    logging.info(f"Already completed      : {already_done}")
    logging.info(f"Remaining this run     : {total_work}")
    logging.info("-" * 60)

    if total_work == 0:
        logging.info("Nothing to do — dataset is complete.")
        return

    # CSVs
    manifest_fields = ["file_path", "class_id", "category", "rendered",
                       "unicode_seq", "font_stem", "tier", "split", "font_type"]
    error_fields    = ["class_id", "rendered", "font_stem",
                       "tier", "split", "font_type", "reason"]

    mf, mw = open_csv_appender(MANIFEST_PATH, manifest_fields,
                                write_header=not MANIFEST_PATH.exists())
    ef, ew = open_csv_appender(ERRORS_PATH,   error_fields,
                                write_header=not ERRORS_PATH.exists())

    app = QApplication.instance() or QApplication(sys.argv)

    success_count      = 0
    error_count        = 0
    consecutive_errors = 0
    current_font_path   = None
    current_font_family = None
    current_font_id     = None
    current_font_type   = None

    for idx, (font_info, cls) in enumerate(work, start=1):

        font_path   = font_info["font_path"]
        font_stem   = font_info["font_stem"]
        tier        = font_info["tier"]
        split       = font_info["split"]
        font_type   = font_info.get("font_type", "unicode")
        class_id    = cls["class_id"]
        rendered    = cls["rendered"]
        category    = cls["category"]
        unicode_seq = cls["unicode_seq"]

        print(
            f"\r[{idx}/{total_work}]  "
            f"✓{success_count}  ✗{error_count}  "
            f"[{'U' if font_type == 'unicode' else 'L'}]  "
            f"{font_stem[:24]:<24}  {class_id} {rendered}   ",
            end="", flush=True
        )

        # Load font if changed
        if font_path != current_font_path:
            if current_font_id is not None:
                QFontDatabase.removeApplicationFont(current_font_id)
            fid = QFontDatabase.addApplicationFont(font_path)
            if fid == -1:
                log_error(ew, class_id, rendered, font_stem, tier, split,
                          font_type, "Failed to load font file")
                error_count += 1; consecutive_errors += 1
                current_font_path = font_path; current_font_family = None
                if check_shutdown(consecutive_errors, error_count,
                                  success_count + error_count): break
                continue
            families = QFontDatabase.applicationFontFamilies(fid)
            if not families:
                log_error(ew, class_id, rendered, font_stem, tier, split,
                          font_type, "No font families found")
                error_count += 1; consecutive_errors += 1
                current_font_path = font_path; current_font_family = None
                if check_shutdown(consecutive_errors, error_count,
                                  success_count + error_count): break
                continue
            current_font_id     = fid
            current_font_family = families[0]
            current_font_path   = font_path
            current_font_type   = font_type
            consecutive_errors  = 0

        if current_font_family is None:
            continue

        # Build render string
        if current_font_type == 'unicode':
            render_text = rendered
        else:
            render_text = build_fm_string(rendered)
            if render_text is None:
                log_error(ew, class_id, rendered, font_stem, tier, split,
                          font_type, "fm_mapping_failed")
                continue

        # Render
        try:
            image = render_and_crop(render_text, current_font_family)
        except Exception as ex:
            log_error(ew, class_id, rendered, font_stem, tier, split,
                      font_type, f"Render exception: {ex}")
            error_count += 1; consecutive_errors += 1
            if check_shutdown(consecutive_errors, error_count,
                              success_count + error_count): break
            continue

        if image is None:
            log_error(ew, class_id, rendered, font_stem, tier, split,
                      font_type, "Blank/tofu render")
            error_count += 1; consecutive_errors += 1
            if check_shutdown(consecutive_errors, error_count,
                              success_count + error_count): break
            continue

        # Save
        try:
            filename  = f"{class_id}_{tier}_{font_stem}.png"
            save_path = folder_map[class_id] / filename
            if not image.save(str(save_path)):
                raise IOError("QImage.save() returned False")
        except Exception as ex:
            log_error(ew, class_id, rendered, font_stem, tier, split,
                      font_type, f"Save exception: {ex}")
            error_count += 1; consecutive_errors += 1
            if check_shutdown(consecutive_errors, error_count,
                              success_count + error_count): break
            continue

        # Manifest
        mw.writerow({
            "file_path":   str(save_path.relative_to(OUTPUT_DIR.parent)),
            "class_id":    class_id,
            "category":    category,
            "rendered":    rendered,
            "unicode_seq": unicode_seq,
            "font_stem":   font_stem,
            "tier":        tier,
            "split":       split,
            "font_type":   font_type,
        })

        success_count     += 1
        consecutive_errors = 0

        if success_count % FLUSH_EVERY == 0:
            mf.flush(); ef.flush()

    print()
    mf.flush(); mf.close()
    ef.flush(); ef.close()

    logging.info("=" * 60)
    logging.info("GENERATION COMPLETE")
    logging.info(f"  Successfully rendered : {success_count}")
    logging.info(f"  Hard errors           : {error_count}")
    logging.info(f"  Already existed       : {already_done}")
    logging.info(f"  Total in dataset      : {success_count + already_done}")
    logging.info(f"  Manifest : {MANIFEST_PATH}")
    logging.info(f"  Errors   : {ERRORS_PATH}")
    logging.info(f"  Log      : {LOG_PATH}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()