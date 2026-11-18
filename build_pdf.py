"""Erzeugt aus Ergebnisse.md und Projektskript.md visuell aufbereitete PDF-Dateien.

Aufruf: python build_pdf.py
Benötigt: markdown, pandas, numpy, matplotlib und Google Chrome (Headless-Druck).
KaTeX wird für die Formeln aus dem CDN geladen (Internetzugang nötig).
"""
import base64
import html as htmllib
import json
import re
import subprocess
import sys
from pathlib import Path

import markdown
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
BUILD = ROOT / "_pdf_build"
CH = BUILD / "charts"
CH.mkdir(parents=True, exist_ok=True)
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

C_T, C_B = "#2563eb", "#f97316"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9ca3af",
    "axes.titleweight": "bold", "axes.titlesize": 10, "axes.labelcolor": "#374151",
})

summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
S = pd.DataFrame(summary["summary"])
S["m"] = S.model.str.split("/").str[-1]
CMP = summary["comparison"]
W = pd.concat([pd.read_csv(RES / f"whisper-{m}_word_metrics.csv") for m in ("tiny", "base")])
W["m"] = W.model.str.split("/").str[-1]
TINY, BASE = "openai/whisper-tiny", "openai/whisper-base"

ORDER = ["Occlusion", "CrossAttn (Alignment-Köpfe)", "IG", "Grad×Δ", "CrossAttn (alle Köpfe)"]
LAB = ["Occlusion", "CrossAttn\n(Alignment-Köpfe)", "IG", "Grad×Δ", "CrossAttn\n(alle Köpfe)"]


def de(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def save(fig, name):
    path = CH / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- Diagramme
def chart_methods():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    specs = [("pointing", "Pointing-Accuracy", 0.049, "{:.2f}"),
             ("mass_in", "Masse im Wortsegment (mass_in)", 0.043, "{:.2f}"),
             ("lift_median", "Lift (Median)", 1.0, "{:.1f}")]
    y, h = np.arange(len(ORDER)), 0.36
    for k, (ax, (col, ttl, chance, fmt)) in enumerate(zip(axs, specs)):
        for i, (m, c) in enumerate((("whisper-tiny", C_T), ("whisper-base", C_B))):
            vals = [S[(S.m == m) & (S.method == o)][col].iloc[0] for o in ORDER]
            bars = ax.barh(y + (i - 0.5) * h, vals, h, color=c, label=m)
            for b, v in zip(bars, vals):
                ax.text(v + max(S[col]) * 0.012, b.get_y() + h / 2, fmt.format(v), va="center", fontsize=7.5)
        ax.axvline(chance, ls="--", color="#6b7280", lw=1, label="Zufallsniveau")
        ax.set_yticks(y)
        ax.set_yticklabels(LAB)
        ax.invert_yaxis()
        ax.set_title(ttl)
        ax.set_xlim(0, max(S[col]) * 1.16)
        if k == 0:
            ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Zeitliche Lokalisierung je Methode (1751 bzw. 1774 Wörter)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return save(fig, "methods.png")


def chart_wordtypes():
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    x, w = np.arange(len(ORDER)), 0.38
    for ax, m in zip(axs, ("whisper-tiny", "whisper-base")):
        for i, (flag, lab, c) in enumerate(((False, "Inhaltswörter", "#0ea5e9"), (True, "Funktionswörter", "#f43f5e"))):
            vals = [W[(W.m == m) & (W.method == o) & (W.func == flag)].hit.mean() for o in ORDER]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, color=c, label=lab)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + w / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)
        ax.axhline(0.049, ls="--", color="#6b7280", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(LAB, fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_title(m)
    axs[0].set_ylabel("Pointing-Accuracy")
    axs[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Trefferquote nach Wortart (gestrichelt: Zufallsniveau 0,05)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return save(fig, "wordtypes.png")


def chart_compare():
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.2))
    panels = [("WER (%)", "WER", 100, "{:.1f}"), ("IG-Pointing", "IG_pointing", 1, "{:.2f}"),
              ("Pointing der\nAlignment-Köpfe", "align_heads_pointing", 1, "{:.2f}"),
              ("Ø Pointing aller\nCross-Köpfe", "mean_cross_head", 1, "{:.2f}"),
              ("Encoder `local`,\nSchicht 0", "enc_local_first", 1, "{:.2f}"),
              ("Encoder `local`,\nletzte Schicht", "enc_local_last", 1, "{:.2f}"),
              ("Deletion-AUC (IG)\nΔ log p", "del_auc_IG", 1, "{:.2f}")]
    for ax, (ttl, key, mult, fmt) in zip(axs.ravel(), panels):
        vals = [CMP[TINY][key] * mult, CMP[BASE][key] * mult]
        bars = ax.bar(["tiny", "base"], vals, color=[C_T, C_B], width=0.6)
        span = max(abs(v) for v in vals)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (span * 0.03 if v >= 0 else -span * 0.03), fmt.format(v),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, fontweight="bold")
        ax.set_title(ttl.replace("`", ""), fontsize=9)
        lo, hi = min(0, min(vals)), max(0, max(vals))
        ax.set_ylim(lo - span * 0.22 if lo < 0 else 0, hi + span * 0.22)
        ax.tick_params(axis="x", length=0)
    ax = axs[1, 3]
    x, w = np.arange(2), 0.36
    ali = [CMP[TINY]["dWER_align_heads"], CMP[BASE]["dWER_align_heads"]]
    rnd = [CMP[TINY]["dWER_random_heads"], CMP[BASE]["dWER_random_heads"]]
    b1 = ax.bar(x - w / 2, ali, w, color="#7c3aed", label="Alignment-Köpfe")
    b2 = ax.bar(x + w / 2, rnd, w, color="#94a3b8", label="Zufallsköpfe")
    for b, v in list(zip(b1, ali)) + list(zip(b2, rnd)):
        ax.text(b.get_x() + w / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["tiny", "base"])
    ax.set_ylim(0, max(ali + rnd) * 1.25)
    ax.set_title("ΔWER bei Ablation", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.tick_params(axis="x", length=0)
    fig.suptitle("whisper-tiny (blau) gegen whisper-base (orange)", fontweight="bold", y=1.0)
    fig.tight_layout()
    return save(fig, "compare.png")


def chart_igerr():
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for m, c, lab in (("whisper-tiny", C_T, "whisper-tiny"), ("whisper-base", C_B, "whisper-base")):
        e = W[(W.m == m) & (W.method == "IG")].ig_err.clip(upper=2.0)
        ax.hist(e, bins=np.linspace(0, 2, 41), alpha=0.55, color=c, label=f"{lab} (Median {de(e.median(), 2)})")
        ax.axvline(e.median(), color=c, lw=1.6)
    ax.axvspan(0, 0.05, color="#10b981", alpha=0.25, label="Zielbereich < 5 %")
    ax.set_xlabel("relativer Completeness-Fehler (über 2,0 abgeschnitten)")
    ax.set_ylabel("Anzahl Wörter")
    ax.set_title("Integrated Gradients: Verteilung des Completeness-Fehlers")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save(fig, "igerr.png")


def chart_ablation():
    data = {
        "whisper-tiny": [("Enc L0 K4", -5.57, "e"), ("Enc L0 K2", -4.99, "e"), ("Dec L1 K2", -2.90, "d"),
                         ("Enc L0 K3", -2.79, "e"), ("Enc L0 K5", -2.20, "e"), ("Dec L0 K4", -1.20, "d")],
        "whisper-base": [("Dec L2 K2", -1.13, "d"), ("Enc L0 K1", -0.24, "e"), ("Dec L3 K0", -0.23, "d"),
                         ("Enc L0 K6", -0.14, "e")],
    }
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[6, 4]))
    for ax, (m, rows) in zip(axs, data.items()):
        rows = sorted(rows, key=lambda r: r[1])
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        cols = ["#4f46e5" if r[2] == "e" else "#0d9488" for r in rows]
        bars = ax.barh(names, vals, color=cols)
        for b, v in zip(bars, vals):
            ax.text(v - 0.08, b.get_y() + b.get_height() / 2, f"{v:.2f}".replace(".", ","), va="center", ha="right", fontsize=8)
        ax.set_xlim(min(vals) * 1.25, 0)
        ax.invert_yaxis()
        ax.set_title(m)
        ax.set_xlabel("Δ mittleres Token-log p bei Ablation des Kopfes")
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(color="#4f46e5", label="Encoder-Self-Attention"),
                           Patch(color="#0d9488", label="Decoder-Cross-Attention")], frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Stärkste Einzelkopf-Ablationen (aus den Heatmaps abgelesen; L = Schicht, K = Kopf)", fontweight="bold", y=1.03)
    fig.tight_layout()
    return save(fig, "ablation.png")


def chart_bands():
    bands = ["37–372", "410–745", "782–1128", "1173–1657", "1722–2433", "2528–3571", "3711–5244", "5449–7699"]
    content = [0.41, 1.12, 0.55, 0.60, 0.46, 0.40, 0.07, 0.12]
    func = [0.23, 0.62, 0.17, 0.27, 0.19, 0.22, 0.02, 0.07]
    x, w = np.arange(len(bands)), 0.38
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    b1 = ax.bar(x - w / 2, content, w, color="#0ea5e9", label="Inhaltswörter")
    b2 = ax.bar(x + w / 2, func, w, color="#f43f5e", label="Funktionswörter")
    for b, v in list(zip(b1, content)) + list(zip(b2, func)):
        ax.text(b.get_x() + w / 2, v + 0.015, f"{v:.2f}".replace(".", ","), ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=8)
    ax.set_xlabel("Frequenzband (Hz)")
    ax.set_ylabel("Abfall von log p")
    ax.set_title("Band-Occlusion (whisper-tiny, aus der Abbildung abgelesen): erster Formant dominiert")
    ax.legend(frameon=False)
    fig.tight_layout()
    return save(fig, "bands.png")


def make_charts():
    return {
        "methods": (chart_methods(), "Lokalisierung je Methode: Pointing, Masse im Wort und Lift (Zufallsniveau gestrichelt)"),
        "wordtypes": (chart_wordtypes(), "Pointing-Accuracy nach Wortart"),
        "compare": (chart_compare(), "Modellvergleich auf einen Blick"),
        "ig_err": (chart_igerr(), "Verteilung des IG-Completeness-Fehlers: nur ein kleiner Teil der Wörter liegt im Zielbereich"),
        "ablation": (chart_ablation(), "Stärkste Einzelkopf-Ablationen"),
        "bands": (chart_bands(), "Band-Occlusion nach Frequenzband und Wortart"),
    }


PNG_CAPTIONS = {
    "example.png": "Beispiel-Äußerung (whisper-tiny): Spektrogramm, IG-Karte, Zeitprofile und Cross-Attention",
    "erp.png": "Ereigniskorreliertes Relevanzprofil (whisper-tiny)",
    "tf1_wordtypes.png": "TF1: Wortarten, Wortdauer und Tokenzahl (whisper-tiny, IG)",
    "tf2_spectral.png": "TF2: Spektrale Analyse (whisper-tiny)",
    "tf3_cross_heads.png": "TF3: Pointing-Accuracy und Padding-Masse der Cross-Attention-Köpfe",
    "tf4_enc_whisper-tiny.png": "TF4: Kennzahlen der Encoder-Köpfe (whisper-tiny)",
    "tf4_enc_whisper-base.png": "TF4: Kennzahlen der Encoder-Köpfe (whisper-base)",
    "tf4_enc_maps.png": "TF4: Encoder-Attention-Karten der Beispiel-Äußerung (whisper-tiny)",
    "tf5_deletion.png": "TF5: Deletion-Kurven",
    "tf5_ablation_whisper-tiny.png": "TF5: Kopf-Ablation (whisper-tiny)",
    "tf5_ablation_whisper-base.png": "TF5: Kopf-Ablation (whisper-base)",
}


# ---------------------------------------------------------------- Hilfsfunktionen
def b64(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def figure(path, caption):
    return f'<figure><img src="{b64(path)}" alt=""><figcaption>{caption}</figcaption></figure>'


def protect_math(text):
    store = []

    def rep(m):
        store.append(m.group(0))
        return f"@@M{len(store) - 1}@@"

    text = re.sub(r"\$\$.+?\$\$", rep, text, flags=re.S)
    text = re.sub(r"(?<!\\)\$(?!\s)[^$\n]+?(?<!\s)\$", rep, text)
    return text, store


def restore_math(h, store):
    for i, s in enumerate(store):
        h = h.replace(f"@@M{i}@@", htmllib.escape(s, quote=False))
    return h


FLOW = """
<div class="flow">
  <div class="fnode">100 Äußerungen aus LibriSpeech <i>test-clean</i> mit MFA-Wortgrenzen</div>
  <div class="farrow">▼</div>
  <div class="fnode">Whisper transkribiert (greedy)</div>
  <div class="farrow">▼</div>
  <div class="fnode">Tokens zu Wörtern, Abgleich mit der Referenz – nur korrekt erkannte Wörter bleiben</div>
  <div class="farrow">▼</div>
  <div class="fnode key">Teacher Forcing: <b>log p pro Wort</b> = Erklärungsziel</div>
  <div class="farrow">▼</div>
  <div class="frow">
    <div class="fnode">Integrated Gradients<br>Gradient×Δ</div>
    <div class="fnode">Zeit-Occlusion<br>Band-Occlusion</div>
    <div class="fnode">Cross-Attention<br>Encoder-Self-Attention</div>
  </div>
  <div class="farrow">▼</div>
  <div class="fnode key">Zeitprofil pro Wort</div>
  <div class="farrow">▼</div>
  <div class="frow">
    <div class="fnode">Vergleich mit der MFA-Wortgrenze<br><small>mass_in · lift · Pointing · Kontext</small></div>
    <div class="fnode">Deletion-Test<br><small>25 Äußerungen</small></div>
    <div class="fnode">Kopf-Ablation<br><small>25 Äußerungen</small></div>
  </div>
  <div class="farrow">▼</div>
  <div class="fnode">Auswertung und Plots</div>
</div>
"""


def kpi_html(items):
    cells = []
    for label, t, b, note, warn in items:
        cells.append(
            f'<div class="kpi{" warnk" if warn else ""}"><div class="kl">{label}</div>'
            f'<div class="kv"><span class="vt"><em>tiny</em>{t}</span><span class="vb"><em>base</em>{b}</span></div>'
            f'<div class="kn">{note}</div></div>')
    return '<div class="kpis">' + "".join(cells) + "</div>"


def section_end_insert(html, heading_regex, fragments):
    for m in re.finditer(r"<h([1-6])\b[^>]*>(.*?)</h\1>", html, flags=re.S):
        inner = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if re.search(heading_regex, inner):
            level = int(m.group(1))
            nxt = re.search(rf"<h[1-{level}]\b", html[m.end():])
            pos = m.end() + nxt.start() if nxt else len(html)
            return html[:pos] + "".join(fragments) + html[pos:]
    print("WARNUNG: Überschrift nicht gefunden:", heading_regex)
    return html


CSS = r"""
:root{--ink:#1f2937;--muted:#6b7280;--primary:#1e3a8a;--accent:#4338ca}
*{box-sizing:border-box}
html{-webkit-print-color-adjust:exact;print-color-adjust:exact}
@page{size:A4;margin:17mm 15mm 18mm 15mm;
  @bottom-left{content:"__FOOTER__";font:8.5pt "Segoe UI",Arial,sans-serif;color:#9ca3af}
  @bottom-right{content:counter(page) " / " counter(pages);font:8.5pt "Segoe UI",Arial,sans-serif;color:#6b7280}}
@page :first{margin:0;@bottom-left{content:none}@bottom-right{content:none}}
body{font-family:"Segoe UI","Helvetica Neue",Arial,sans-serif;font-size:10pt;line-height:1.52;color:var(--ink);margin:0}
.cover{width:210mm;height:296mm;position:relative;overflow:hidden;break-after:page}
.banner{height:112mm;padding:27mm 20mm 0 20mm;color:#fff;
  background:radial-gradient(circle at 86% 18%,rgba(255,255,255,.24),transparent 36%),
  radial-gradient(circle at 8% 112%,rgba(249,115,22,.6),transparent 46%),
  linear-gradient(135deg,#1e3a8a 0%,#4338ca 55%,#7c3aed 100%)}
.kicker{letter-spacing:.2em;text-transform:uppercase;font-size:8.5pt;opacity:.85}
.banner h1{font-size:35pt;line-height:1.08;margin:8mm 0 4mm;font-weight:800;color:#fff;border:0;padding:0}
.sub{font-size:13pt;opacity:.93;max-width:155mm}
.chips{margin-top:9mm}
.chips span{display:inline-block;background:rgba(255,255,255,.17);border:1px solid rgba(255,255,255,.38);border-radius:999px;padding:2px 12px;margin:0 6px 6px 0;font-size:9pt}
.toc{padding:13mm 20mm 0 20mm}
.toc-title{font-size:15pt;color:var(--primary);margin:0 0 5mm;padding:0 0 2mm;border-bottom:2px solid #c7d2fe;break-before:auto}
.toc ol{column-count:2;column-gap:12mm;margin:0;padding:0;list-style:none}
.toc li{break-inside:avoid;margin:0 0 7px;font-size:10pt;padding-left:3mm;border-left:3px solid #c7d2fe}
.toc a{color:var(--ink);text-decoration:none}
.doc>h2{font-size:20pt;line-height:1.2;color:var(--primary);margin:11mm 0 5mm;padding:0 0 2mm 4mm;border-left:6px solid #6366f1;border-bottom:1px solid #e0e7ff;break-after:avoid}
.pbreak .doc>h2{break-before:page;margin-top:0}
.pbreak .doc>h2:first-child{break-before:auto}
h3{font-size:13.5pt;color:var(--accent);margin:7mm 0 2.5mm;break-after:avoid}
h3::before{content:"";display:inline-block;width:8px;height:8px;border-radius:2px;background:#7c3aed;margin-right:7px;vertical-align:1px}
h4{font-size:11.5pt;color:#6d28d9;margin:5mm 0 2mm;break-after:avoid}
p{margin:0 0 7px}
a{color:var(--accent);text-decoration:none}.ref{font-weight:600;color:var(--accent)}
strong{color:#111827}
ul,ol{padding-left:1.35em;margin:3px 0 9px}li{margin:2px 0}
code{font-family:"Cascadia Mono",Consolas,monospace;font-size:8.6pt;background:#eef2ff;color:#3730a3;padding:0 4px;border-radius:4px}
pre{background:#0f172a;color:#e2e8f0;padding:10px 12px;border-radius:8px;font-size:8.5pt;overflow:hidden;break-inside:avoid}
pre code{background:none;color:inherit;padding:0}
hr{display:none}
table{border-collapse:separate;border-spacing:0;width:100%;margin:8px 0 14px;font-size:8.7pt;border:1px solid #dbe0ee;border-radius:7px;overflow:hidden}
thead{display:table-header-group}
th{background:linear-gradient(135deg,#1e40af,#4f46e5);color:#fff;text-align:left;padding:6px 8px;font-weight:600}
td{padding:5px 8px;border-top:1px solid #eceff7;vertical-align:top}
tr{break-inside:avoid}
tbody tr:nth-child(even) td{background:#f5f7ff}
td:first-child{font-weight:600}
p.callout{border-left:4px solid;padding:8px 13px;border-radius:0 7px 7px 0;margin:10px 0}
p.warn{background:#fffbeb;border-color:#f59e0b}p.ok{background:#ecfdf5;border-color:#10b981}p.info{background:#eff6ff;border-color:#3b82f6}
blockquote{margin:10px 0;padding:8px 14px;border-left:4px solid #6366f1;background:#eef2ff;border-radius:0 7px 7px 0}
blockquote p{margin:0}
.badge{display:inline-block;padding:1px 9px;border-radius:999px;font-weight:700;font-size:8.3pt;color:#fff}
.badge.green{background:#059669}.badge.amber{background:#d97706}.badge.red{background:#dc2626}
figure{margin:12px 0 16px;text-align:center;break-inside:avoid}
figure img{max-width:100%;max-height:232mm;border:1px solid #e5e7eb;border-radius:7px;box-shadow:0 2px 8px rgba(15,23,42,.08)}
figcaption{font-size:8.5pt;color:var(--muted);margin-top:5px}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin:4mm 0 8mm;break-inside:avoid}
.kpi{border:1px solid #dbe0ee;border-radius:10px;padding:9px 11px;background:linear-gradient(180deg,#fff,#f5f7ff)}
.kpi.warnk{background:linear-gradient(180deg,#fff,#fffbeb);border-color:#fcd34d}
.kl{font-size:8.5pt;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.kv{display:flex;gap:14px;margin:3px 0 2px;font-size:17pt;font-weight:800}
.kv em{display:block;font-style:normal;font-size:7pt;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
.vt{color:#2563eb}.vb{color:#ea580c}
.kn{font-size:8pt;color:var(--muted)}
.flow{margin:10px auto 14px;max-width:170mm;text-align:center;break-inside:avoid}
.fnode{background:#eef2ff;border:1.5px solid #6366f1;color:#1e1b4b;border-radius:9px;padding:6px 12px;font-size:9pt;display:inline-block;min-width:62%}
.fnode.key{background:#ede9fe;border-color:#7c3aed}
.fnode small{color:#4b5563}
.frow{display:flex;gap:8px;justify-content:center}.frow .fnode{flex:1;min-width:0}
.farrow{color:#6366f1;font-size:11pt;line-height:1.15;margin:1px 0}
"""

MATH_HEAD = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],throwOnError:false})"></script>
"""


def build(md_name, out_name, subtitle, footer, page_break, charts, injections, kpis=None, drop_toc=False):
    text = (ROOT / md_name).read_text(encoding="utf-8")
    text = re.sub(r"\A# .*\n", "", text)
    if drop_toc:
        text = re.sub(r"## Inhaltsverzeichnis.*?\n---\n", "", text, flags=re.S)
    text = re.sub(r"```mermaid.*?```", "@@FLOW@@", text, flags=re.S)
    text = re.sub(r"(?m)^  (?=(?:\d+\.|[*\-])\s)", "    ", text)
    text, store = protect_math(text)
    h = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    h = restore_math(h, store)
    h = h.replace("<p>@@FLOW@@</p>", FLOW)

    # Verweise auf .md/.ipynb sind im PDF nicht anklickbar
    h = re.sub(r'<a href="[^"]*\.(?:md|ipynb)">(.*?)</a>', r'<span class="ref">\1</span>', h)

    # PNG-Verweise (Ergebnisse.md) werden als Abbildungen eingebettet
    def png_paragraph(m):
        body = m.group(1)
        hrefs = re.findall(r'href="(results/[^"]+\.png)"', body)
        if not hrefs:
            return m.group(0)
        body = re.sub(r'<a href="results/[^"]+\.png">(.*?)</a>', r'<span class="ref">\1</span>', body)
        figs = "".join(figure(ROOT / p, PNG_CAPTIONS.get(Path(p).name, Path(p).name)) for p in hrefs)
        return f"<p>{body}</p>{figs}"

    h = re.sub(r"<p>((?:(?!</p>).)*?href=\"results/[^\"]+\.png\"(?:(?!</p>).)*?)</p>", png_paragraph, h, flags=re.S)

    # Hervorhebungen
    h = h.replace("<p><strong>Befund:</strong>", '<p class="callout warn"><strong>Befund:</strong>')
    h = h.replace("<p><strong>Kernaussage", '<p class="callout ok"><strong>Kernaussage')
    h = h.replace("<p><strong>Fazit der Ablation", '<p class="callout info"><strong>Fazit der Ablation')

    def badge(m):
        t = m.group(1)
        cls = "amber" if re.match(r"Teilweise|Im Mittel|Gemischt", t) else "red" if t.startswith("Nicht") else "green"
        return f'<td><span class="badge {cls}">{t}</span>'

    h = re.sub(r"<td><strong>(Bestätigt|Teilweise bestätigt|Im Mittel bestätigt|Gemischt|Nicht bestätigt)</strong>", badge, h)

    # Diagramme und Abbildungen an Abschnittsenden
    for heading_regex, items in injections:
        frags = []
        for it in items:
            if it.startswith("results/"):
                frags.append(figure(ROOT / it, PNG_CAPTIONS[Path(it).name]))
            else:
                p, cap = charts[it]
                frags.append(figure(p, cap))
        h = section_end_insert(h, heading_regex, frags)

    if kpis:
        h = section_end_insert_before(h, kpis)

    toc = "".join(f'<li><a href="#{i}">{t}</a></li>' for i, t in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', h))
    title = "Wo hört Whisper hin?"
    cover = f"""
<section class="cover">
  <div class="banner">
    <div class="kicker">Erklärbare KI · Spracherkennung · Whisper</div>
    <h1>{title}</h1>
    <div class="sub">{subtitle}</div>
    <div class="chips"><span>whisper-tiny</span><span>whisper-base</span><span>LibriSpeech test-clean</span><span>100 Äußerungen</span><span>16. November 2026</span></div>
  </div>
  <div class="toc"><h2 class="toc-title">Inhalt</h2><ol>{toc}</ol></div>
</section>"""
    doc = f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><title>{title} – {footer}</title>
<style>{CSS.replace("__FOOTER__", f"{title} · {footer}")}</style>{MATH_HEAD}</head>
<body class="{"pbreak" if page_break else ""}">{cover}<main class="doc">{h}</main></body></html>"""
    html_path = BUILD / out_name.replace(".pdf", ".html")
    html_path.write_text(doc, encoding="utf-8")
    pdf_path = ROOT / out_name
    cmd = [str(CHROME), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
           "--virtual-time-budget=30000", f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    print("OK:", pdf_path, round(pdf_path.stat().st_size / 1e6, 1), "MB")


def section_end_insert_before(html, fragment):
    m = re.search(r"<h2\b[^>]*>1 Versuchsaufbau", html)
    return html[:m.start()] + fragment + html[m.start():] if m else html


if __name__ == "__main__":
    charts = make_charts()
    kp = kpi_html([
        ("Wortfehlerrate (WER)", f"{de(CMP[TINY]['WER'] * 100, 1)} %", f"{de(CMP[BASE]['WER'] * 100, 1)} %", "kleiner = besser", False),
        ("IG-Pointing", f"{de(CMP[TINY]['IG_pointing'] * 100, 1)} %", f"{de(CMP[BASE]['IG_pointing'] * 100, 1)} %", "Zufallsniveau 4,9 %", False),
        ("Occlusion-Pointing",
         f"{de(S[(S.model == TINY) & (S.method == 'Occlusion')].pointing.iloc[0] * 100, 1)} %",
         f"{de(S[(S.model == BASE) & (S.method == 'Occlusion')].pointing.iloc[0] * 100, 1)} %", "schärfste Lokalisierung", False),
        ("Pointing der Alignment-Köpfe", de(CMP[TINY]["align_heads_pointing"]), de(CMP[BASE]["align_heads_pointing"]),
         f"Ø aller Köpfe: {de(CMP[TINY]['mean_cross_head'])} / {de(CMP[BASE]['mean_cross_head'])}", False),
        ("Bester Cross-Attention-Kopf", de(CMP[TINY]["best_cross_head"]), de(CMP[BASE]["best_cross_head"]), "in den letzten Decoder-Schichten", False),
        ("IG-Completeness-Fehler (Median)",
         f"{de(W[(W.m == 'whisper-tiny') & (W.method == 'IG')].ig_err.median() * 100, 0)} %",
         f"{de(W[(W.m == 'whisper-base') & (W.method == 'IG')].ig_err.median() * 100, 0)} %", "Ziel: unter 5 % – IG nicht konvergiert", True),
    ])

    build("Ergebnisse.md", "Ergebnisse.pdf", "Auswertung aller Ergebnisse – Lokalisierung der Evidenz in Whisper", "Ergebnisse",
          page_break=False, charts=charts, kpis=kp, injections=[
              (r"^2 Qualität der Integrated Gradients", ["ig_err"]),
              (r"^3 Gesamtübersicht", ["compare"]),
              (r"^4\.1 ", ["methods"]),
              (r"^4\.5 ", ["wordtypes"]),
              (r"^5\.2 ", ["bands"]),
              (r"^8\.2 ", ["ablation"]),
          ])

    build("Projektskript.md", "Projektskript.pdf", "Projektskript – Hintergrund, Code und Ergebnisse zum Nachvollziehen", "Projektskript",
          page_break=True, charts=charts, drop_toc=True, injections=[
              (r"^8\.2 ", ["ig_err"]),
              (r"^8\.3 ", ["methods", "wordtypes", "results/erp.png", "results/example.png", "results/tf1_wordtypes.png"]),
              (r"^8\.4 ", ["results/tf2_spectral.png", "bands"]),
              (r"^8\.5 ", ["results/tf3_cross_heads.png"]),
              (r"^8\.6 ", ["results/tf4_enc_whisper-tiny.png", "results/tf4_enc_whisper-base.png", "results/tf4_enc_maps.png"]),
              (r"^Deletion$", ["results/tf5_deletion.png"]),
              (r"^Ablation einzelner Köpfe$", ["ablation", "results/tf5_ablation_whisper-tiny.png", "results/tf5_ablation_whisper-base.png"]),
              (r"^8\.8 ", ["compare"]),
          ])
