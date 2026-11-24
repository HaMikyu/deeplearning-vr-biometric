"""Drukuje tabele markdown z results/ do wklejenia w README / raport.

Uruchom po: train_*.py, ablation.py i rsa.py.
"""

import csv
import json
from pathlib import Path

RESULTS = Path("results")
NAMES = {"resnet": "ResNet1D", "gru": "GRUClassifier", "transformer": "TransformerClassifier"}


def read_csv(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main_models():
    print("### Trening glowny (modalnosc full, seed 42)\n")
    print("| Model | Parametry | Best epoch | Val Rank-1 | Test Rank-1 | Test Rank-5 | Test EER |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for model, name in NAMES.items():
        path = RESULTS / f"{model}_metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        t = m["test"]
        print(
            f"| {name} | {m['parameters']:,} | {m['best_epoch']} | {m['val']['rank1']:.4f} "
            f"| {t['rank1']:.4f} | {t['rank5']:.4f} | {t['eer']:.4f} |"
        )
    print()


def ablation():
    rows = read_csv(RESULTS / "ablation.csv")
    n = rows[0]["n_seeds"]
    print(f"### Ablacja (srednia +- odchylenie z {n} seedow, dzien 2)\n")
    cols = [
        ("rank1", "Rank-1"),
        ("rank5", "Rank-5"),
        ("eer", "EER"),
        ("cos_genuine", "cos genuine"),
        ("cos_impostor", "cos impostor"),
        ("rank1_quest", "Rank-1 Quest"),
        ("rank1_vive", "Rank-1 Vive"),
        ("rank1_cosmos", "Rank-1 Cosmos"),
    ]
    print("| Model | Modalnosc | " + " | ".join(label for _, label in cols) + " |")
    print("|---|---|" + "---:|" * len(cols))
    for row in rows:
        cells = [
            f"{float(row[f'{key}_mean']):.3f} ± {float(row[f'{key}_std']):.3f}" for key, _ in cols
        ]
        print(f"| {NAMES[row['model']]} | {row['modality']} | " + " | ".join(cells) + " |")
    print()


def rsa():
    rows = read_csv(RESULTS / "rsa_reference.csv")
    print("### RSA z tozsamoscia i urzadzeniem (seed 42, dzien 2)\n")
    print("| Przebieg | RSA z tozsamoscia | RSA z urzadzeniem |")
    print("|---|---:|---:|")
    for row in rows:
        print(f"| {row['run']} | {float(row['rsa_identity']):.3f} | {float(row['rsa_device']):.3f} |")
    print()


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # znak ± na konsoli Windows
    main_models()
    ablation()
    rsa()
