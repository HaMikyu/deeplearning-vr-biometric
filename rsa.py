"""RSA na embeddingach testowych (dzien 2). Najpierw uruchom ablation.py.

RDM = 1 - cosine miedzy probkami. Porownujemy RDM-y modeli korelacja
Spearmana oraz kazdy RDM z RDM-em tozsamosci (0 = ta sama osoba) i
urzadzenia (0 = to samo urzadzenie).
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from ablation import MODALITIES, MODELS
from metrics import cosine_similarity, load_embedding


def main():
    runs = [f"{model}_{modality}" for model in MODELS for modality in MODALITIES]
    upper = None
    rdms = {}
    for run in runs:
        emb, labels, devices = load_embedding(
            Path("embeddings") / run / "test_embedding.npz"
        )
        if upper is None:
            upper = np.triu_indices(len(labels), k=1)
            identity = (labels[:, None] != labels[None, :])[upper]
            device = (devices[:, None] != devices[None, :])[upper]
        rdms[run] = (1.0 - cosine_similarity(emb, emb))[upper]

    n = len(runs)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = spearmanr(rdms[runs[i]], rdms[runs[j]])[0]

    with open("results/rsa_models.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run"] + runs)
        for run, row in zip(runs, matrix):
            writer.writerow([run] + [f"{value:.4f}" for value in row])

    with open("results/rsa_reference.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "rsa_identity", "rsa_device"])
        for run in runs:
            r_identity = spearmanr(rdms[run], identity)[0]
            r_device = spearmanr(rdms[run], device)[0]
            writer.writerow([run, f"{r_identity:.4f}", f"{r_device:.4f}"])
            print(f"{run:22s} identity={r_identity:.3f} device={r_device:.3f}")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(n), runs, rotation=45, ha="right")
    ax.set_yticks(range(n), runs)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title("RSA model-model (Spearman RDM, dzien 2)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig("results/rsa_models.png", dpi=130)


if __name__ == "__main__":
    main()
