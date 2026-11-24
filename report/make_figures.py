"""Wykresy do raportu (report/figures/*.pdf) liczone wylacznie z results/ i embeddings/.

Uruchom z katalogu glownego repo:  python report/make_figures.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics import cosine_similarity, identity_scores  # noqa: E402

RESULTS = ROOT / "results"
EMB = ROOT / "embeddings"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Paleta kategoryczna (stala kolejnosc, kolor przypisany do modelu, nie do rangi)
COLORS = {
    "resnet": "#2a78d6",
    "gru": "#eb6834",
    "transformer": "#1baf7a",
    "inception": "#eda100",
    "ensemble": "#e87ba4",
}
NAMES = {
    "resnet": "ResNet1D",
    "gru": "GRU",
    "transformer": "Transformer",
    "inception": "InceptionTime",
    "ensemble": "Ensemble",
}
BASE_MODELS = ["resnet", "gru", "transformer"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9893"
GRID = "#e6e5e1"
NEUTRAL = "#b9b7b0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def read_csv(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=170)
    plt.close(fig)
    print("zapisano", name)


def grouped_bars(ax, groups, series, values, errors=None, ylim=None, fmt="{:.3f}", label_values=True):
    """values[s][g]; seria = model (kolor modelu), grupa = os x."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        pos = x - 0.4 + width * (i + 0.5)
        vals = np.array(values[s])
        err = None if errors is None else np.array(errors[s])
        ax.bar(pos, vals - (ylim[0] if ylim else 0), width * 0.88, bottom=ylim[0] if ylim else 0,
               color=COLORS[s], label=NAMES[s], zorder=2)
        if err is not None:
            ax.errorbar(pos, vals, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5, zorder=3)
        if label_values:
            for p, v, e in zip(pos, vals, err if err is not None else [0] * len(vals)):
                ax.text(p, v + e + 0.004, fmt.format(v), ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(x, groups)
    if ylim:
        ax.set_ylim(*ylim)


def load_emb(name, split):
    data = np.load(EMB / name / f"{split}_embedding.npz")
    return data["embeddings"], data["labels"], data["device_ids"]


def cmc(name, max_k=10):
    g, gy, _ = load_emb(name, "train")
    q, qy, _ = load_emb(name, "test")
    scores, ids = identity_scores(cosine_similarity(q, g), gy)
    order = ids[np.argsort(-scores, axis=1)]
    rank = (order == qy[:, None]).argmax(axis=1) + 1
    return np.array([(rank <= k).mean() for k in range(1, max_k + 1)])


def genuine_impostor(name):
    g, gy, _ = load_emb(name, "train")
    q, qy, _ = load_emb(name, "test")
    sim = cosine_similarity(q, g)
    same = qy[:, None] == gy[None, :]
    return sim[same], sim[~same]


def det_curve(gen, imp):
    thresholds = np.linspace(-1, 1, 801)
    far = np.array([(imp >= t).mean() for t in thresholds])
    frr = np.array([(gen < t).mean() for t in thresholds])
    i = np.argmin(np.abs(far - frr))
    return far, frr, (far[i] + frr[i]) / 2


# ---------------------------------------------------------------- baseline

def fig_ablation():
    rows = read_csv(RESULTS / "ablation.csv")
    mods = ["position", "rotation", "full"]
    get = lambda m, mod, k: float(next(r for r in rows if r["model"] == m and r["modality"] == mod)[k])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    vals = {m: [get(m, mod, "rank1_mean") for mod in mods] for m in BASE_MODELS}
    errs = {m: [get(m, mod, "rank1_std") for mod in mods] for m in BASE_MODELS}
    grouped_bars(axes[0], ["pozycja", "rotacja (6D)", "pełny"], BASE_MODELS, vals, errs, ylim=(0.5, 1.0))
    axes[0].set_title("Rank-1 (dzień 2), średnia ± odch. std. z 3 seedów", loc="left")
    axes[0].set_ylabel("Rank-1")
    vals = {m: [get(m, mod, "eer_mean") for mod in mods] for m in BASE_MODELS}
    errs = {m: [get(m, mod, "eer_std") for mod in mods] for m in BASE_MODELS}
    grouped_bars(axes[1], ["pozycja", "rotacja (6D)", "pełny"], BASE_MODELS, vals, errs, ylim=(0, 0.2))
    axes[1].set_title("EER (niżej = lepiej)", loc="left")
    axes[1].set_ylabel("EER")
    axes[0].legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    save(fig, "ablation_modality")


def fig_devices():
    rows = read_csv(RESULTS / "ablation.csv")
    devs = [("quest", "Quest"), ("vive", "Vive"), ("cosmos", "Cosmos")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, (mod, title) in zip(axes, [("position", "pozycja"), ("rotation", "rotacja (6D)"), ("full", "pełny")]):
        sel = {r["model"]: r for r in rows if r["modality"] == mod}
        vals = {m: [float(sel[m][f"rank1_{d}_mean"]) for d, _ in devs] for m in BASE_MODELS}
        errs = {m: [float(sel[m][f"rank1_{d}_std"]) for d, _ in devs] for m in BASE_MODELS}
        grouped_bars(ax, [t for _, t in devs], BASE_MODELS, vals, errs, ylim=(0.4, 1.0), label_values=False)
        ax.set_title(f"Modalność: {title}", loc="left")
    axes[0].set_ylabel("Rank-1 (dzień 2)")
    axes[0].legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.tight_layout()
    save(fig, "rank1_devices")


def fig_training_curves():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    for ax, m in zip(axes, BASE_MODELS):
        rows = read_csv(RESULTS / f"{m}_history.csv")
        ep = [int(r["epoch"]) for r in rows]
        ax.plot(ep, [float(r["train_loss"]) for r in rows], color=NEUTRAL, lw=1.6, label="train loss")
        ax.plot(ep, [float(r["val_loss"]) for r in rows], color=COLORS[m], lw=1.6, label="val loss")
        best = json.loads((RESULTS / f"{m}_metrics.json").read_text(encoding="utf-8"))["best_epoch"]
        ax.axvline(best, color=MUTED, lw=0.8, ls="--")
        ax.text(best - 1, 2.0, "najlepsza\nepoka " + str(best), fontsize=6.5, color=INK2, ha="right")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.4, 0.6, 1, 2, 3]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(NAMES[m], loc="left")
        ax.set_xlabel("epoka")
    axes[0].set_ylabel("cross-entropy (skala log.)")
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.75))
    fig.tight_layout()
    save(fig, "training_curves")


def fig_cmc(runs, name, title):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ks = np.arange(1, 11)
    for run, color_key, label in runs:
        curve = cmc(run)
        ax.plot(ks, curve, color=COLORS[color_key], lw=1.8, marker="o", ms=3.5, label=f"{label} (R1={curve[0]:.3f})")
    ax.set_xticks(ks)
    ax.set_xlabel("k (pozycja w rankingu osób)")
    ax.set_ylabel("Rank-k (dzień 2)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, name)


def fig_scores(runs, name):
    fig, axes = plt.subplots(2, len(runs), figsize=(7.2, 4.4))
    bins = np.linspace(-1, 1, 81)
    for i, (run, color_key, label) in enumerate(runs):
        gen, imp = genuine_impostor(run)
        ax = axes[0, i]
        ax.hist(imp, bins=bins, density=True, color=NEUTRAL, label="impostor", zorder=2)
        ax.hist(gen, bins=bins, density=True, color=COLORS[color_key], alpha=0.85, label="genuine", zorder=3)
        ax.set_title(label, loc="left")
        ax.set_xlabel("podobieństwo cosinusowe" if len(runs) <= 3 else "cos")
        ax.set_yticks([])
        if i == 0:
            ax.legend(loc="upper left")
        far, frr, eer = det_curve(gen, imp)
        ax = axes[1, i]
        ok = (far > 0) & (frr > 0)
        ax.loglog(far[ok], frr[ok], color=COLORS[color_key], lw=1.8)
        ax.plot([eer], [eer], "o", ms=5, color=COLORS[color_key], mec="white", mew=1.5, zorder=4)
        ax.plot([1e-4, 1], [1e-4, 1], color=GRID, lw=0.8, zorder=1)
        ax.text(0.97, 0.95, f"EER = {eer:.3f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=INK)
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel("FAR")
        ax.grid(axis="both", color=GRID, lw=0.6, which="major")
        if i == 0:
            ax.set_ylabel("FRR")
    fig.tight_layout()
    save(fig, name)


def fig_rsa():
    rows = read_csv(RESULTS / "rsa_models.csv")
    runs = [r["run"] for r in rows]
    mat = np.array([[float(r[c]) for c in runs] for r in rows])
    labels = [f"{NAMES[r.split('_')[0]]} {dict(position='poz.', rotation='rot.', full='pełny')[r.split('_')[1]]}" for r in runs]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9cc0ee", "#2a78d6", "#123f78"])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(runs)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(runs)), labels)
    ax.grid(False)
    for i in range(len(runs)):
        for j in range(len(runs)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if mat[i, j] > 0.55 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("korelacja Spearmana RDM")
    ax.set_title("RSA model–model (embeddingi dnia 2)", loc="left")
    fig.tight_layout()
    save(fig, "rsa_models")


def fig_rsa_reference():
    rows = read_csv(RESULTS / "rsa_reference.csv")
    mods = ["position", "rotation", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for ax, key, title, ylim in [
        (axes[0], "rsa_identity", "RSA z tożsamością (wyżej = lepiej)", (0.15, 0.28)),
        (axes[1], "rsa_device", "RSA z urządzeniem (niżej = mniej biasu)", (0, 0.1)),
    ]:
        vals = {m: [float(next(r for r in rows if r["run"] == f"{m}_{mod}")[key]) for mod in mods] for m in BASE_MODELS}
        grouped_bars(ax, ["pozycja", "rotacja (6D)", "pełny"], BASE_MODELS, vals, ylim=ylim)
        ax.set_title(title, loc="left")
    axes[0].legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.tight_layout()
    save(fig, "rsa_reference")


# ---------------------------------------------------------------- SOTA

SOTA_STEPS = [
    ("base", "baseline\n(CE)"),
    ("arcface", "+ ArcFace"),
    ("arc_aug", "+ augm."),
    ("arc_aug_ema", "+ EMA"),
    ("arc_aug_ema_center", "+ centr.\nx/z"),
    ("full", "+ prędkość\n(pełny)"),
]


def screen_rows():
    rows = []
    for path in RESULTS.glob("sota_screen_*.csv"):
        rows += read_csv(path)
    return rows


def fig_sota_components():
    rows = screen_rows()
    if not rows:
        return
    models = [m for m in ["resnet", "gru", "transformer", "inception"] if any(r["model"] == m for r in rows)]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    x = np.arange(len(SOTA_STEPS))
    for m in models:
        for ax, key in [(axes[0], "dev_rank1"), (axes[1], "dev_eer")]:
            ys = []
            for variant, _ in SOTA_STEPS:
                vals = [float(r[key]) for r in rows if r["model"] == m and r["variant"] == variant]
                ys.append(np.mean(vals) if vals else np.nan)
            ax.plot(x, ys, color=COLORS[m], lw=1.8, marker="o", ms=4, label=NAMES[m])
    for ax, title in [(axes[0], "Rank-1 (dev: dzień 2, rzuty 0–4), średnia z 2 seedów"),
                      (axes[1], "EER (dev, niżej = lepiej)")]:
        ax.set_title(title, loc="left")
    axes[1].set_xticks(x, [s for _, s in SOTA_STEPS], fontsize=7.5)
    axes[0].legend(ncol=4, loc="lower right")
    fig.tight_layout()
    save(fig, "sota_components")


def fig_sota_vs_baseline(variant):
    path = RESULTS / f"sota_summary_{variant}.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))["runs"]
    base = {r["model"]: r for r in read_csv(RESULTS / "ablation.csv") if r["modality"] == "full"}
    order = [m for m in ["resnet", "gru", "transformer", "inception"] if m in summary]
    ens = [k for k in summary if k.startswith("ensemble_")]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6))
    for ax, key, title in [(axes[0], "rank1", "Rank-1 (cały dzień 2)"), (axes[1], "eer", "EER (cały dzień 2, niżej = lepiej)")]:
        labels, xs = [], 0
        for m in order + ens:
            color = COLORS["ensemble" if m.startswith("ensemble_") else m]
            short = {"transformer": "Transf.", "inception": "Inception\nTime"}
            if m.startswith("ensemble_"):
                label = "Ens.\n4 modele" if m.count("+") == 3 else "Ens.\nbez GRU"
            else:
                label = short.get(m, NAMES[m])
            if m in base:
                v, e = float(base[m][f"{key}_mean"]), float(base[m][f"{key}_std"])
                ax.bar(xs - 0.2, v, 0.38, color=NEUTRAL, zorder=2)
                ax.errorbar(xs - 0.2, v, yerr=e, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
                ax.text(xs - 0.2, v + e, f"{v:.3f}", ha="center", va="bottom", fontsize=6, color=INK2)
            s = summary[m][f"all_{key}"]
            ax.bar(xs + 0.2, s["mean"], 0.38, color=color, zorder=2)
            ax.errorbar(xs + 0.2, s["mean"], yerr=s["std"], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
            ax.text(xs + 0.2, s["mean"] + s["std"], f"{s['mean']:.3f}", ha="center", va="bottom", fontsize=6, color=INK2)
            labels.append(label)
            xs += 1
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(title, loc="left")
    axes[0].set_ylim(0.75, 1.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=NEUTRAL), plt.Rectangle((0, 0), 1, 1, color=INK2)]
    axes[1].legend(handles, ["baseline S2 (pełny, 3 seedy)", "S4 (kolor modelu, 3 seedy)"], ncol=1,
                   loc="upper right")
    fig.tight_layout()
    save(fig, "sota_vs_baseline")


def main():
    fig_ablation()
    fig_devices()
    fig_training_curves()
    fig_rsa()
    fig_rsa_reference()
    base_runs = [(m, m, NAMES[m]) for m in BASE_MODELS]
    fig_cmc(base_runs, "cmc_baseline", "CMC – S1 (pełny, seed 42)")
    fig_scores(base_runs, "scores_baseline")
    fig_sota_components()
    variant = sys.argv[1] if len(sys.argv) > 1 else "full_day1"
    fig_sota_vs_baseline(variant)
    sota_runs = [(f"sota_{m}", m, NAMES[m]) for m in ["resnet", "gru", "transformer", "inception"]
                 if (EMB / f"sota_{m}").exists()]
    ens = sorted(p.name for p in EMB.glob("sota_ensemble_*"))
    # ensemble wybrany na dev: wszystkie 4 modele (najdluzsza nazwa)
    sota_runs += [(e, "ensemble", "Ensemble") for e in sorted(ens, key=len)[-1:]]
    if sota_runs:
        fig_cmc(sota_runs, "cmc_sota", "CMC – S4 (seed 42)")
        fig_scores(sota_runs, "scores_sota")


if __name__ == "__main__":
    main()
