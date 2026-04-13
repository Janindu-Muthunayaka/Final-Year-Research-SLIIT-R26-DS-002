# =============================================================================
# stage5_reporting.py  —  HTML Reports, CSV Files & Composite PNGs
#
# Generates all human-readable output from classification results:
#   per-image HTML report, composite PNG, segment detail CSV,
#   summary CSV, master summary, batch report.
#
# INPUT  (from stage4 / stage3 via run_stage5_reporting):
#   - stems     : list[str]        — image stems to generate reports for
#   - work_root : str              — base working directory (contains temp/)
#   - elapsed_s : float            — total pipeline elapsed time in seconds
#   - cfg       : PipelineConfig   — tunable parameters
#
# OUTPUT (from run_stage5_reporting):
#   - list[dict]  — per-image summary dicts, each containing:
#       stem           : str   — image stem
#       fname          : str   — original filename
#       predicted_text : str   — predicted Sinhala text
#       ground_truth   : str   — label text
#       n_aksharas     : int   — number of recognised characters
#       n_words        : int   — number of words
#       wer            : float — Word Error Rate (%)
#       cer            : float — Character Error Rate (%)
#       html_path      : str   — path to generated HTML report
# =============================================================================

from __future__ import annotations

import os
import json
import csv
import shutil
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

from stage1_config import (
    PipelineConfig,
    _DEVICE, INFER_BATCH_SIZE, S1_WORKERS, CUDNN_BENCHMARK,
)
from stage4_classification import _class_to_sinhala

# =============================================================================
# COMPOSITE PNG
# =============================================================================

def _try_font(size: int = 13):
    for name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _build_composite_png(aksharas: list,
                          char_canvas_size: int,
                          out_path: str) -> None:
    """Build a grid PNG showing all akshara crops with labels."""
    if not aksharas:
        return
    cell_w = cell_h_img = char_canvas_size
    cell_h  = cell_h_img + 46
    n       = len(aksharas)
    cols    = min(n, 10)
    rows    = (n + cols - 1) // cols
    comp    = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 245))
    draw    = ImageDraw.Draw(comp)
    font    = _try_font(13)

    for i, ak in enumerate(aksharas):
        col, row = i % cols, i // cols
        ox, oy   = col * cell_w, row * cell_h
        if not os.path.exists(ak["crop_path"]):
            continue
        crop_img = Image.open(ak["crop_path"]).resize((cell_w, cell_h_img))
        comp.paste(crop_img, (ox, oy))
        label = f"{ak['predicted_char']}  {ak['confidence']:.1f}%"
        draw.rectangle([ox, oy + cell_h_img, ox + cell_w, oy + cell_h],
                        fill=(20, 20, 30))
        draw.text((ox + 4, oy + cell_h_img + 4),
                  label, fill=(255, 255, 255), font=font)
        draw.text((ox + 4, oy + cell_h_img + 22),
                  ak["chosen_by"], fill=(100, 200, 150), font=font)
    comp.save(out_path)

# =============================================================================
# HTML
# =============================================================================

def _cards_html(aksharas: list) -> str:
    html = ""
    for ak in aksharas:
        preds      = ak["predictions"]
        sinhala    = ak["predicted_char"]
        border_col = "#4ecca3" if ak["window_segs"] > 1 else "#334"
        crop_rel   = os.path.basename(ak["crop_path"])

        alt_rows = "".join(
            f'<tr><td class="si">{_class_to_sinhala(p[0])}</td>'
            f'<td>{p[1]:.1f}%</td>'
            f'<td><div class="bar" style="width:{min(p[1],100):.0f}px"></div></td>'
            f'</tr>' for p in preds
        )
        html += f"""
        <div class="card" style="border-color:{border_col}">
          <img src="{crop_rel}" alt="akshara {ak['index']}">
          <div class="top si">{sinhala}</div>
          <div class="conf">{ak['confidence']:.1f}%</div>
          <div class="chosen">{ak['chosen_by']}</div>
          <table class="alts"><tr><th>Char</th><th>Conf</th><th></th></tr>
          {alt_rows}</table>
        </div>"""
    return html


def _build_html(stem, orig_path, skel_path, composite_path,
                aksharas, ground_truth, predicted_text,
                wer, cer, out_path,
                tess_text=None, tess_wer=None, tess_cer=None) -> None:
    def rel(p): return os.path.basename(p)
    pred_html   = "".join(
        f'<span class="pc" title="{ak["chosen_by"]}">{ak["predicted_char"]}</span>'
        for ak in aksharas)
    cards       = _cards_html(aksharas)
    gt_display  = ground_truth if ground_truth else "— (not available)"
    wer_col     = "#4ecca3" if wer < 30 else ("#f0a500" if wer < 60 else "#e94560")
    cer_col     = "#4ecca3" if cer < 30 else ("#f0a500" if cer < 60 else "#e94560")
    wer_display = f"{wer}%" if ground_truth else "N/A"
    cer_display = f"{cer}%" if ground_truth else "N/A"

    html = f"""<!DOCTYPE html><html lang="si"><head><meta charset="UTF-8">
<title>OCR — {stem}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:sans-serif;background:#0f0f1a;color:#e0e0e0;padding:24px}}
h1{{color:#e94560;font-size:20px;margin-bottom:4px}}
.meta{{color:#555;font-size:12px;margin-bottom:20px}}
.tab-bar{{display:flex;gap:0;margin-bottom:0;border-bottom:2px solid #1a2a50}}
.tab-btn{{padding:9px 22px;background:#16213e;border:1px solid #1a2a50;border-bottom:none;
  cursor:pointer;color:#888;font-size:13px;border-radius:6px 6px 0 0;margin-right:4px}}
.tab-btn.active{{background:#0f0f1a;color:#e94560;border-color:#e94560;
  border-bottom:2px solid #0f0f1a;margin-bottom:-2px}}
.tab-content{{display:none;padding-top:18px}}
.tab-content.active{{display:block}}
.panel{{background:#16213e;border:1px solid #1a2a50;border-radius:10px;padding:16px;margin-bottom:18px}}
.panel h2{{font-size:12px;color:#888;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px}}
.imgs{{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}}
.imgs img{{max-height:90px;border-radius:4px;background:#fff;border:1px solid #333}}
.imgs .lb{{font-size:11px;color:#666;margin-top:3px;text-align:center}}
.metrics{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:14px}}
.metric{{background:#0f1a30;border-radius:8px;padding:10px 18px;text-align:center;min-width:110px}}
.metric .v{{font-size:26px;font-weight:bold}}
.metric .l{{font-size:11px;color:#888;margin-top:2px}}
.cmp{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.tb{{background:#0f1a30;border-radius:8px;padding:12px}}
.tb .lb{{font-size:11px;color:#888;margin-bottom:6px}}
.si{{font-size:20px;line-height:1.7}}
.pc{{display:inline-block;margin:2px;padding:2px 5px;background:#1a2a50;border-radius:4px;
  font-size:18px;cursor:default}}
.pc:hover{{background:#e94560;color:#fff}}
.final-answer{{background:#0f1a30;border-radius:12px;padding:24px;text-align:center;margin-bottom:18px}}
.final-answer .fa-text{{font-size:48px;line-height:1.8;letter-spacing:6px;color:#4ecca3}}
.final-answer .fa-label{{font-size:11px;color:#555;margin-bottom:10px}}
.gt-box{{background:#0f1a30;border-radius:12px;padding:18px;text-align:center}}
    .gt-box .gt-text{{font-size:36px;line-height:1.8;color:#aaa}}
    .gt-box .gt-label{{font-size:11px;color:#555;margin-bottom:8px}}
    .comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-top: 20px; }}
    .comp-box {{ background: #0f1a30; border-radius: 12px; padding: 18px; text-align: center; border: 1px solid #1a2a40; }}
    .comp-box .val {{ font-size: 32px; line-height: 1.6; color: #aaa; }}
    .comp-box .lab {{ font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
    .comp-box .met {{ font-size: 12px; color: #4ecca3; margin-top: 8px; }}
    .legend{{font-size:11px;color:#666;margin-bottom:12px}}
.leg{{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle;margin-right:4px}}
.grid{{display:flex;flex-wrap:wrap;gap:12px}}
.card{{background:#16213e;border:1px solid #334;border-radius:8px;padding:10px;width:160px}}
.card img{{width:100%;border-radius:4px;background:#fff;display:block}}
.top{{text-align:center;font-size:26px;margin:6px 0 2px;color:#e94560}}
.conf{{text-align:center;font-size:12px;color:#4ecca3;margin-bottom:2px}}
.chosen{{text-align:center;font-size:10px;color:#555;margin-bottom:6px}}
.alts{{width:100%;border-collapse:collapse;font-size:11px}}
.alts th{{color:#555;text-align:left;padding:2px 3px;border-bottom:1px solid #1a2a50}}
.alts td{{padding:2px 3px}}
.bar{{height:5px;background:#4ecca3;border-radius:3px}}
.model-badge{{display:inline-block;background:#1a2a50;color:#4ecca3;font-size:10px;
  padding:2px 8px;border-radius:4px;margin-left:8px}}
</style></head><body>
<h1>Sinhala OCR — {stem}<span class="model-badge">EfficientNetV2-S</span></h1>
<div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Aksharas: {len(aksharas)}</div>
<div class="tab-bar">
  <button class="tab-btn active" onclick="showTab('final')">📄 Final Answer</button>
  <button class="tab-btn" onclick="showTab('segments')">🔬 Segment Detail</button>
</div>
<div id="tab-final" class="tab-content active">
  <div class="panel"><h2>Input images</h2>
    <div class="imgs">
      <div><img src="{rel(orig_path)}"><div class="lb">Original</div></div>
      <div><img src="{rel(skel_path)}"><div class="lb">Skeleton</div></div>
      <div><img src="{rel(composite_path)}"><div class="lb">All aksharas</div></div>
    </div>
  </div>
  <div class="panel"><h2>Accuracy metrics</h2>
    <div class="metrics">
      <div class="metric"><div class="v" style="color:{wer_col}">{wer_display}</div><div class="l">WER</div></div>
      <div class="metric"><div class="v" style="color:{cer_col}">{cer_display}</div><div class="l">CER</div></div>
      <div class="metric"><div class="v" style="color:#aaa">{len(aksharas)}</div><div class="l">Aksharas</div></div>
    </div>
  </div>
    <div class="panel"><h2>Predicted text</h2>
      <div class="final-answer">
        <div class="fa-label">RECOGNISED OUTPUT (Proposed Pipeline)</div>
        <div class="fa-text si">{predicted_text if predicted_text else "—"}</div>
      </div>
      
      <div class="comparison-grid">
        <div class="comp-box">
          <div class="lab">Ground Truth</div>
          <div class="val si">{gt_display}</div>
        </div>
        <div class="comp-box">
          <div class="lab">Tesseract Prediction</div>
          <div class="val si">{tess_text if tess_text else "—"}</div>
          <div class="met">CER: {tess_cer if tess_cer is not None else "N/A"}% | WER: {tess_wer if tess_wer is not None else "N/A"}%</div>
        </div>
        <div class="comp-box">
          <div class="lab">Ours (Dynamic)</div>
          <div class="val si" style="color:#4ecca3">{predicted_text if predicted_text else "—"}</div>
          <div class="met">CER: {cer}% | WER: {wer}%</div>
        </div>
      </div>
    </div>
</div>
<div id="tab-segments" class="tab-content">
  <div class="panel"><h2>Per-character predictions</h2>
    <div class="metrics">
      <div class="metric"><div class="v" style="color:{wer_col}">{wer_display}</div><div class="l">WER</div></div>
      <div class="metric"><div class="v" style="color:{cer_col}">{cer_display}</div><div class="l">CER</div></div>
    </div>
    <div class="cmp">
      <div class="tb"><div class="lb">GROUND TRUTH</div><div class="si">{gt_display}</div></div>
      <div class="tb"><div class="lb">PREDICTED</div><div>{pred_html}</div></div>
    </div>
  </div>
  <div class="panel"><h2>Akshara cards</h2>
    <div class="legend">
      <span class="leg" style="background:#4ecca3;border:1px solid #4ecca3"></span>multi-seg &nbsp;
      <span class="leg" style="background:#334;border:1px solid #555"></span>1-seg
    </div>
    <div class="grid">{cards}</div>
  </div>
</div>
<script>
function showTab(name){{
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  event.target.classList.add('active');
}}
</script></body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

# =============================================================================
# CSV WRITERS
# =============================================================================

def _write_segments_csv(aksharas, ground_truth, wer, cer, out_path):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as cf:
        writer = csv.writer(cf)
        header = ["Index", "Predicted_Char", "Top_Class", "Confidence_%",
                  "Chosen_By", "Seg_Start", "Seg_End", "Window_Segs",
                  "X_Start_px", "X_End_px", "Word_Index"]
        for k in range(1, 5):
            header += [f"Alt{k}_Char", f"Alt{k}_Class", f"Alt{k}_Conf_%"]
        header += ["Ground_Truth", "WER_%", "CER_%"]
        writer.writerow(header)
        for ak in aksharas:
            preds = ak["predictions"]
            row   = [ak["index"], ak["predicted_char"], preds[0][0],
                     preds[0][1], ak["chosen_by"], ak["seg_start"],
                     ak["seg_end"], ak["window_segs"], ak["x_start"],
                     ak["x_end"], ak.get("word_index", "")]
            for p in preds[1:5]:
                row += [_class_to_sinhala(p[0]), p[0], p[1]]
            while len(row) < 11 + 4 * 3:
                row.append("")
            row += ([ground_truth, wer, cer] if ak["index"] == 0 else ["", "", ""])
            writer.writerow(row)


def _write_summary_csv(stem, fname, predicted_text,
                        ground_truth, n_aksharas, wer, cer, out_path,
                        tess_text=None, tess_wer=None, tess_cer=None):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as cf:
        writer = csv.writer(cf)
        writer.writerow(["Image", "Stem", "Predicted_Text", "Ground_Truth",
                         "N_Aksharas", "WER_%", "CER_%", 
                         "Tesseract_Text", "Tesseract_WER_%", "Tesseract_CER_%"])
        writer.writerow([fname, stem, predicted_text, ground_truth,
                         n_aksharas, wer, cer,
                         tess_text, tess_wer, tess_cer])

# =============================================================================
# PROCESS ONE IMAGE
# =============================================================================

def _process_one(stem: str,
                 work_root: str,
                 results_dir: str,
                 cfg: PipelineConfig,
                 tess_data: dict = None) -> dict:
    """Generate HTML, CSV, and composite PNG for one classified image."""
    temp_dir     = os.path.join(work_root, "temp", stem)
    meta_path    = os.path.join(temp_dir, "meta.json")
    results_path = os.path.join(temp_dir, "results.json")

    with open(meta_path,    "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(results_path, "r", encoding="utf-8") as f:
        res  = json.load(f)

    fname          = meta["fname"]
    ground_truth   = res["ground_truth"]
    predicted_text = res["predicted_text"]
    aksharas       = res["aksharas"]
    wer            = res["wer"]
    cer            = res["cer"]

    tess_text = tess_data.get("text") if tess_data else None
    tess_wer  = tess_data.get("wer")  if tess_data else None
    tess_cer  = tess_data.get("cer")  if tess_data else None

    img_out = os.path.join(results_dir, stem)
    os.makedirs(img_out, exist_ok=True)

    orig_dest = os.path.join(img_out, fname)
    skel_dest = os.path.join(img_out, os.path.basename(meta["skel_path"]))
    if not os.path.exists(orig_dest):
        shutil.copy2(meta["orig_path"], orig_dest)
    if not os.path.exists(skel_dest):
        shutil.copy2(meta["skel_path"], skel_dest)

    for ak in aksharas:
        src_crop = ak["crop_path"]
        if not os.path.exists(src_crop):
            continue
        crop_dest = os.path.join(img_out, os.path.basename(src_crop))
        if not os.path.exists(crop_dest):
            shutil.copy2(src_crop, crop_dest)
        ak["crop_path"] = crop_dest

    composite_path = os.path.join(img_out, "composite.png")
    _build_composite_png(aksharas, cfg.char_canvas_size, composite_path)

    html_path = os.path.join(img_out, "index.html")
    _build_html(stem=stem, orig_path=orig_dest, skel_path=skel_dest,
                composite_path=composite_path, aksharas=aksharas,
                ground_truth=ground_truth, predicted_text=predicted_text,
                wer=wer, cer=cer, out_path=html_path,
                tess_text=tess_text, tess_wer=tess_wer, tess_cer=tess_cer)

    _write_segments_csv(aksharas=aksharas, ground_truth=ground_truth,
                         wer=wer, cer=cer,
                         out_path=os.path.join(img_out,
                                                f"{stem}_segments_detail.csv"))
    _write_summary_csv(stem=stem, fname=fname, predicted_text=predicted_text,
                        ground_truth=ground_truth, n_aksharas=len(aksharas),
                        wer=wer, cer=cer,
                        out_path=os.path.join(img_out,
                                              f"{stem}_final_summary.csv"),
                        tess_text=tess_text, tess_wer=tess_wer, tess_cer=tess_cer)

    print(f"    {stem}  ->  {img_out}")
    return {
        "stem":           stem,
        "fname":          fname,
        "predicted_text": predicted_text,
        "ground_truth":   ground_truth,
        "n_aksharas":     len(aksharas),
        "n_words":        res.get("n_words") or 0,
        "wer":            wer,
        "cer":            cer,
        "html_path":      html_path,
        "tess_text":      tess_text,
        "tess_wer":       tess_wer,
        "tess_cer":       tess_cer,
    }

# =============================================================================
# BATCH REPORTS
# =============================================================================

def _write_master_summary(all_results: list, results_dir: str) -> None:
    out_path = os.path.join(results_dir, "master_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as mf:
        writer = csv.writer(mf)
        writer.writerow(["Image", "Stem", "Predicted_Text", "Ground_Truth",
                         "N_Aksharas", "WER_%", "CER_%", 
                         "Tess_Text", "Tess_WER_%", "Tess_CER_%"])
        for s in all_results:
            writer.writerow([s["fname"], s["stem"], s["predicted_text"],
                             s["ground_truth"], s["n_aksharas"],
                             s["wer"], s["cer"],
                             s.get("tess_text"), s.get("tess_wer"), s.get("tess_cer")])
    print(f"    master_summary.csv -> {out_path}")


def _write_batch_report(all_results: list,
                         results_dir: str,
                         elapsed_s: float,
                         cfg: PipelineConfig) -> None:
    gt_results = [s for s in all_results if s["ground_truth"]]
    mean_wer   = (sum(s["wer"] for s in gt_results) / len(gt_results)
                  if gt_results else None)
    mean_cer   = (sum(s["cer"] for s in gt_results) / len(gt_results)
                  if gt_results else None)

    tess_results = [s for s in gt_results if s.get("tess_cer") is not None]
    tess_mean_wer = (sum(s["tess_wer"] for s in tess_results) / len(tess_results)
                     if tess_results else None)
    tess_mean_cer = (sum(s["tess_cer"] for s in tess_results) / len(tess_results)
                     if tess_results else None)

    lines = [
        "=" * 72,
        "  Sinhala OCR  —  Batch Report  (EfficientNetV2-S)",
        f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Word spacer: {'ENABLED' if cfg.word_spacer_enabled else 'DISABLED'}",
        f"  Device     : {_DEVICE}",
        "=" * 72,
        f"  Total images : {len(all_results)}",
        f"  With GT      : {len(gt_results)}",
        f"  Elapsed      : {elapsed_s:.1f}s  "
        f"({elapsed_s / max(len(all_results), 1):.1f}s/image)",
        "",
        "  ── Parameters used ──────────────────────────────────────────",
    ]
    for k, v in cfg.as_param_dict().items():
        lines.append(f"    {k:<24} = {v}")
    lines += [
        "",
        "  ── GPU / HW config ──────────────────────────────────────────",
        f"    INFER_BATCH_SIZE   = {INFER_BATCH_SIZE}",
        f"    S1_WORKERS         = {S1_WORKERS}",
        f"    CUDNN_BENCHMARK    = {CUDNN_BENCHMARK}",
        "",
        "  ── Accuracy ──────────────────────────────────────────────────",
        f"  Mean WER (Ours) : {mean_wer:.2f}%" if mean_wer is not None else "  Mean WER (Ours) : N/A",
        f"  Mean CER (Ours) : {mean_cer:.2f}%" if mean_cer is not None else "  Mean CER (Ours) : N/A",
        f"  Mean WER (Tess) : {tess_mean_wer:.2f}%" if tess_mean_wer is not None else "  Mean WER (Tess) : N/A",
        f"  Mean CER (Tess) : {tess_mean_cer:.2f}%" if tess_mean_cer is not None else "  Mean CER (Tess) : N/A",
        "",
        "  ── Per-image ─────────────────────────────────────────────────",
        f"  {'Filename':<42} {'WER (Ours)':>12} {'CER (Ours)':>12} {'CER (Tess)':>12}",
        "  " + "─" * 70,
    ]
    for s in all_results:
        wer_str  = f"{s['wer']:.2f}%" if s["ground_truth"] else "N/A"
        cer_str  = f"{s['cer']:.2f}%" if s["ground_truth"] else "N/A"
        tcer_str = f"{s.get('tess_cer', 0.0):.2f}%" if s["ground_truth"] and s.get("tess_cer") is not None else "N/A"
        lines.append(f"  {s['fname']:<42} {wer_str:>12} {cer_str:>12} {tcer_str:>12}")
    lines += ["", "=" * 72]

    out_path = os.path.join(results_dir, "batch_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"    batch_report.txt -> {out_path}")

# =============================================================================
# RUN STAGE 5 — REPORTING
# =============================================================================

def run_stage5_reporting(cfg: PipelineConfig,
                         stems: list,
                         work_root: str,
                         elapsed_s: float) -> list:
    """Generate HTML/CSV/PNG reports for all classified images."""
    import time as _time
    t0          = _time.time()
    results_dir = os.path.join(work_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"  [Stage 5 - Reporting] Building results for {len(stems)} image(s)...\n")
    all_results = []
    for i, stem in enumerate(stems, 1):
        print(f"    [{i}/{len(stems)}]", end=" ")
        try:
            summary = _process_one(stem, work_root, results_dir, cfg)
            all_results.append(summary)
        except Exception as exc:
            import traceback
            print(f"\n           ERROR: {exc}")
            traceback.print_exc()

    print("\n  [Stage 5 – Reporting] Writing master files …")
    _write_master_summary(all_results, results_dir)
    total_elapsed = elapsed_s + (_time.time() - t0)
    _write_batch_report(all_results, results_dir, total_elapsed, cfg)
    print(f"  [Stage 5 - Reporting] Done - results in: {results_dir}\n")
    return all_results


def generate_flat_report(stem: str,
                         work_root: str,
                         out_dir: str,
                         cfg: PipelineConfig,
                         tess_data: dict = None) -> dict:
    """
    Generate everything for one image directly into out_dir (flat).
    No 'results/stem' or 'temp/stem' subfolders added automatically.
    This is used by Part 3 (dynamic pipeline) for cleaner output.
    """
    temp_dir     = os.path.join(work_root, "temp", stem)
    meta_path    = os.path.join(temp_dir, "meta.json")
    results_path = os.path.join(temp_dir, "results.json")

    if not os.path.exists(results_path):
        return {}

    with open(meta_path,    "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(results_path, "r", encoding="utf-8") as f:
        res  = json.load(f)

    fname          = meta["fname"]
    ground_truth   = res["ground_truth"]
    predicted_text = res["predicted_text"]
    aksharas       = res["aksharas"]
    wer            = res["wer"]
    cer            = res["cer"]

    tess_text = tess_data.get("text") if tess_data else None
    tess_wer  = tess_data.get("wer")  if tess_data else None
    tess_cer  = tess_data.get("cer")  if tess_data else None

    os.makedirs(out_dir, exist_ok=True)

    # 1. Copy images to flat out_dir
    orig_dest   = os.path.join(out_dir, fname)
    skel_dest   = os.path.join(out_dir, "skeleton_final.png")
    binary_dest = os.path.join(out_dir, "binary_final.png")

    if not os.path.exists(orig_dest):
        shutil.copy2(meta["orig_path"], orig_dest)
    if not os.path.exists(skel_dest):
        shutil.copy2(meta["skel_path"], skel_dest)
    if not os.path.exists(binary_dest):
        shutil.copy2(meta["binary_path"], binary_dest)

    # 2. Copy crops
    for ak in aksharas:
        src_crop = ak["crop_path"]
        if not os.path.exists(src_crop):
            continue
        crop_dest = os.path.join(out_dir, os.path.basename(src_crop))
        if not os.path.exists(crop_dest):
            shutil.copy2(src_crop, crop_dest)
        ak["crop_path"] = crop_dest

    # 3. Build composite
    composite_path = os.path.join(out_dir, "aksharas_grid.png")
    _build_composite_png(aksharas, cfg.char_canvas_size, composite_path)

    # 4. Build HTML
    html_path = os.path.join(out_dir, "index.html")
    _build_html(stem=stem, orig_path=orig_dest, skel_path=skel_dest,
                composite_path=composite_path, aksharas=aksharas,
                ground_truth=ground_truth, predicted_text=predicted_text,
                wer=wer, cer=cer, out_path=html_path,
                tess_text=tess_text, tess_wer=tess_wer, tess_cer=tess_cer)

    # 5. Build CSVs
    _write_segments_csv(aksharas=aksharas, ground_truth=ground_truth,
                         wer=wer, cer=cer,
                         out_path=os.path.join(out_dir, "segments_detail.csv"))
    _write_summary_csv(stem=stem, fname=fname, predicted_text=predicted_text,
                        ground_truth=ground_truth, n_aksharas=len(aksharas),
                        wer=wer, cer=cer,
                        out_path=os.path.join(out_dir, "image_summary.csv"),
                        tess_text=tess_text, tess_wer=tess_wer, tess_cer=tess_cer)

    return {
        "html_path":      html_path,
        "predicted_text": predicted_text,
        "cer":            cer,
        "wer":            wer,
        "tess_text":      tess_text,
        "tess_cer":       tess_cer,
        "tess_wer":       tess_wer,
    }
