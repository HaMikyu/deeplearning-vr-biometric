"""Ablacja: model x modalnosc (position / rotation / full), cross-day.

Zapisuje results/ablation.csv oraz embeddingi w embeddings/<model>_<modalnosc>/.
"""

import csv
from pathlib import Path

from metrics import evaluate
from train_common import run_training, save_embedding

MODELS = ["resnet", "gru", "transformer"]
MODALITIES = ["position", "rotation", "full"]


def main():
    rows = []
    for model_name in MODELS:
        for modality in MODALITIES:
            result = run_training(
                model_name,
                dashboard_port=None,
                save_artifacts=False,
                verbose=False,
                modality=modality,
                return_embeddings=True,
            )
            emb = result["embeddings"]
            for split in ("train", "val", "test"):
                save_embedding(
                    Path("embeddings") / f"{model_name}_{modality}" / f"{split}_embedding.npz",
                    emb[split],
                )
            row = {
                "model": model_name,
                "modality": modality,
                **evaluate(
                    emb["test"]["embeddings"],
                    emb["test"]["labels"],
                    emb["test"]["device_ids"],
                    emb["train"]["embeddings"],
                    emb["train"]["labels"],
                ),
            }
            rows.append(row)
            print(row, flush=True)

    with open("results/ablation.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
