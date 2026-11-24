"""Tabele LaTeX do raportu (report/tables/*.tex) liczone wylacznie z results/.

Uruchom z katalogu glownego repo:  python report/make_tables.py [wariant_sota]
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = {"resnet": "ResNet1D", "gru": "GRU", "transformer": "Transformer", "inception": "InceptionTime"}
MODS = {"position": "pozycja", "rotation": "rotacja 6D", "full": "pełny"}
BASE = ["resnet", "gru", "transformer"]


def read_csv(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(name, text):
    (OUT / f"{name}.tex").write_text(text, encoding="utf-8")
    print("zapisano", name)


def pm(mean, std, digits=3, bold=False):
    body = f"{mean:.{digits}f} \\pm {std:.{digits}f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def table_hparams():
    rows = []
    for m in BASE:
        b = json.loads((RESULTS / f"best_{m}.json").read_text(encoding="utf-8"))
        t, mo = b["training"], b["model"]
        metrics = json.loads((RESULTS / f"{m}_metrics.json").read_text(encoding="utf-8"))
        if m == "resnet":
            arch = f"kanały bazowe {mo['base_channels']}, 3 etapy $\\times$ 2 bloki"
        elif m == "gru":
            arch = f"BiGRU, {mo['num_layers']} warstwa, hidden {mo['hidden_size']}"
        else:
            arch = (f"$d_\\text{{model}}={mo['d_model']}$, {mo['nhead']} głowy, "
                    f"{mo['num_layers']} warstwy, FFN {mo['dim_feedforward']}")
        rows.append(
            f"{NAMES[m]} & {arch} & {mo['embedding_dim']} & {mo['dropout']:.3f} & "
            f"\\num{{{t['lr']:.1e}}} & \\num{{{t['weight_decay']:.1e}}} & {t['batch_size']} & "
            f"\\num{{{metrics['parameters']}}} \\\\"
        )
    write("hparams", "\n".join(rows))


def table_main():
    rows = []
    best_r1 = max(json.loads((RESULTS / f"{m}_metrics.json").read_text(encoding="utf-8"))["test"]["rank1"] for m in BASE)
    for m in BASE:
        d = json.loads((RESULTS / f"{m}_metrics.json").read_text(encoding="utf-8"))
        t = d["test"]
        b = "\\bfseries " if t["rank1"] == best_r1 else ""
        rows.append(
            f"{b}{NAMES[m]} & {d['best_epoch']} & {d['val']['rank1']:.4f} & {b}{t['rank1']:.4f} & {t['rank5']:.4f} & "
            f"{t['eer']:.4f} & {t['cos_genuine']:.3f} & {t['cos_impostor']:.3f} & "
            f"{t['rank1_quest']:.3f} & {t['rank1_vive']:.3f} & {t['rank1_cosmos']:.3f} \\\\"
        )
    write("main", "\n".join(rows))


def table_ablation():
    rows = read_csv(RESULTS / "ablation.csv")
    out = []
    for m in BASE:
        group = [r for r in rows if r["model"] == m]
        best = max(float(r["rank1_mean"]) for r in group)
        for i, mod in enumerate(["position", "rotation", "full"]):
            r = next(r for r in group if r["modality"] == mod)
            f = lambda k, d=3: pm(float(r[f"{k}_mean"]), float(r[f"{k}_std"]), d)
            first = f"\\multirow{{3}}{{*}}{{{NAMES[m]}}}" if i == 0 else ""
            out.append(
                f"{first} & {MODS[mod]} & "
                f"{pm(float(r['rank1_mean']), float(r['rank1_std']), 3, float(r['rank1_mean']) == best)} & "
                f"{f('rank5')} & {f('eer')} & {float(r['cos_genuine_mean']):.3f} & {float(r['cos_impostor_mean']):.3f} & "
                f"{float(r['rank1_quest_mean']):.3f} & {float(r['rank1_vive_mean']):.3f} & {float(r['rank1_cosmos_mean']):.3f} \\\\"
            )
        out.append("\\midrule" if m != BASE[-1] else "")
    write("ablation", "\n".join(out))


def table_rsa():
    rows = read_csv(RESULTS / "rsa_reference.csv")
    out = []
    for m in BASE:
        cells = []
        for mod in ["position", "rotation", "full"]:
            r = next(r for r in rows if r["run"] == f"{m}_{mod}")
            cells.append(f"{float(r['rsa_identity']):.3f} & {float(r['rsa_device']):.3f}")
        out.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")
    write("rsa", "\n".join(out))


def screen_rows():
    rows = []
    for path in RESULTS.glob("sota_screen_*.csv"):
        rows += read_csv(path)
    return rows


STEPS = [
    ("base", "baseline (CE)"),
    ("arcface", "+ ArcFace"),
    ("arc_aug", "+ augmentacje"),
    ("arc_aug_ema", "+ EMA wag"),
    ("arc_aug_ema_center", "+ centrowanie $x,z$"),
    ("full", "+ prędkości (pełny)"),
]

EXTRAS = [
    ("full_ce", "pełny, ale CE zamiast ArcFace"),
    ("full_noaug", "pełny bez augmentacji"),
    ("full_nocenter", "pełny bez centrowania x/z"),
    ("full_m05", "pełny, margines $m=0{,}5$"),
    ("full_s64", "pełny, skala $s=64$"),
    ("full_e250", "pełny, 250 epok"),
    ("full_day1", "pełny, trening na całym dniu 1"),
]


def mean_of(rows, model, variant, key):
    vals = [float(r[key]) for r in rows if r["model"] == model and r["variant"] == variant]
    return (np.mean(vals), len(vals)) if vals else (None, 0)


def table_sota_components():
    rows = screen_rows()
    if not rows:
        return
    models = [m for m in ["resnet", "gru", "transformer", "inception"]]
    out = []
    for variant, label in STEPS:
        cells = []
        for m in models:
            r1, n = mean_of(rows, m, variant, "dev_rank1")
            eer, _ = mean_of(rows, m, variant, "dev_eer")
            cells.append("--- & ---" if r1 is None else f"{r1:.3f} & {eer:.3f}")
        out.append(f"{label} & " + " & ".join(cells) + " \\\\")
    write("sota_components", "\n".join(out))

    out = []
    for variant, label in [("full", "pełny przepis (odniesienie)")] + EXTRAS:
        cells = []
        for m in ["gru", "transformer"]:
            r1, n = mean_of(rows, m, variant, "dev_rank1")
            eer, _ = mean_of(rows, m, variant, "dev_eer")
            cells.append("--- & ---" if r1 is None else f"{r1:.3f} & {eer:.3f}")
        out.append(f"{label} & " + " & ".join(cells) + " \\\\")
    write("sota_extras", "\n".join(out))


def table_sota_final(variant):
    path = RESULTS / f"sota_summary_{variant}.json"
    if not path.exists():
        return
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    base = {r["model"]: r for r in read_csv(RESULTS / "ablation.csv") if r["modality"] == "full"}
    best_test = max(v["test_rank1"]["mean"] for v in runs.values())
    out = []
    for run, s in runs.items():
        if run.startswith("ensemble_"):
            parts = run.replace("ensemble_", "").split("+")
            name = f"Ensemble ({len(parts)} modele)" if len(parts) == 4 else "Ensemble (bez GRU)"
        else:
            name = NAMES[run]
        b = "\\bfseries " if s["test_rank1"]["mean"] == best_test else ""
        g = lambda k: pm(s[k]["mean"], s[k]["std"], 3, bool(b) and k == "test_rank1")
        if run in base:
            ref = f"{float(base[run]['rank1_mean']):.3f}"
        else:
            ref = "---"
        out.append(
            f"{b}{name} & {ref} & {g('all_rank1')} & {g('test_rank1')} & {g('test_rank5')} & "
            f"{g('test_eer')} & {s['test_rank1_quest']['mean']:.3f} & {s['test_rank1_vive']['mean']:.3f} & "
            f"{s['test_rank1_cosmos']['mean']:.3f} \\\\"
        )
    write("sota_final", "\n".join(out))


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "full_day1"
    table_hparams()
    table_main()
    table_ablation()
    table_rsa()
    table_sota_components()
    table_sota_final(variant)
