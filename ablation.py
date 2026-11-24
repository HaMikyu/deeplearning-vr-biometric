"""Ablacja: model x modalnosc (position / rotation / full) x seed, cross-day.

Zapisuje:
- results/ablation_runs.csv - wynik kazdego przebiegu (model, modalnosc, seed),
- results/ablation.csv      - srednia i odchylenie standardowe po seedach,
- embeddings/<model>_<modalnosc>/{train,val,test}_embedding.npz dla pierwszego
  seeda (uzywane przez rsa.py).

Seed zmienia inicjalizacje, kolejnosc batchy i losowy podzial walidacji.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from configs import get_config
from metrics import evaluate
from train_common import run_training, save_embedding

MODELS = ["resnet", "gru", "transformer"]
MODALITIES = ["position", "rotation", "full"]
DEFAULT_SEEDS = [42, 43, 44]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    metric_keys = [
        key for key in rows[0] if key not in ("model", "modality", "seed", "best_epoch")
    ]
    summary = []
    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["modality"]), []).append(row)
    for (model_name, modality), group in groups.items():
        out = {"model": model_name, "modality": modality, "n_seeds": len(group)}
        for key in metric_keys:
            values = np.array([row[key] for row in group], dtype=float)
            out[f"{key}_mean"] = float(values.mean())
            out[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--modalities", nargs="+", default=MODALITIES, choices=MODALITIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--epochs", type=int, default=None, help="nadpisuje liczbe epok (np. do szybkiego testu)"
    )
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    rows = []
    for model_name in args.models:
        for modality in args.modalities:
            for seed_index, seed in enumerate(args.seeds):
                config = get_config(model_name)
                config["training"]["seed"] = seed
                if args.epochs is not None:
                    config["training"]["epochs"] = args.epochs
                result = run_training(
                    model_name,
                    config,
                    dashboard_port=None,
                    save_artifacts=False,
                    verbose=False,
                    modality=modality,
                    return_embeddings=True,
                )
                emb = result["embeddings"]
                if seed_index == 0:
                    for split in ("train", "val", "test"):
                        save_embedding(
                            Path("embeddings")
                            / f"{model_name}_{modality}"
                            / f"{split}_embedding.npz",
                            emb[split],
                        )
                row = {
                    "model": model_name,
                    "modality": modality,
                    "seed": seed,
                    "best_epoch": result["best_epoch"],
                    **evaluate(
                        emb["test"]["embeddings"],
                        emb["test"]["labels"],
                        emb["test"]["device_ids"],
                        emb["train"]["embeddings"],
                        emb["train"]["labels"],
                    ),
                }
                rows.append(row)
                print(
                    f"{model_name:12s} {modality:9s} seed={seed} "
                    f"epoch={row['best_epoch']:3d} rank1={row['rank1']:.4f} "
                    f"rank5={row['rank5']:.4f} eer={row['eer']:.4f}",
                    flush=True,
                )
                # zapis po kazdym przebiegu, zeby przerwanie nie gubilo wynikow
                write_csv("results/ablation_runs.csv", rows)

    write_csv("results/ablation.csv", summarize(rows))


if __name__ == "__main__":
    main()
