import argparse
import csv
import json
import math
import random
from copy import deepcopy
from pathlib import Path

from configs import get_config
from train_common import run_training


def loguniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_training(rng, model_name):
    return {
        "lr": loguniform(rng, 2e-4, 3e-3),
        "weight_decay": loguniform(rng, 1e-4, 2e-2),
        "batch_size": rng.choice([32, 64]),
    }


def sample_model(rng, model_name):
    if model_name == "resnet":
        return {
            "embedding_dim": rng.choice([64, 128, 256]),
            "dropout": rng.uniform(0.0, 0.3),
            "base_channels": rng.choice([48, 64, 96]),
        }
    if model_name == "gru":
        return {
            "embedding_dim": rng.choice([64, 128, 256]),
            "dropout": rng.uniform(0.0, 0.3),
            "hidden_size": rng.choice([64, 96, 128]),
            "num_layers": rng.choice([1, 2]),
        }
    if model_name == "transformer":
        d_model = rng.choice([64, 96, 128])
        nhead = 8 if d_model == 128 else 4
        return {
            "embedding_dim": rng.choice([64, 128, 256]),
            "dropout": rng.uniform(0.0, 0.25),
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": rng.choice([2, 3]),
            "dim_feedforward": d_model * rng.choice([2, 4]),
        }
    raise ValueError(f"Nieznany model: {model_name}")


def tune_model(model_name, trials, epochs, seed):
    rng = random.Random(seed + sum(ord(ch) for ch in model_name))
    base = get_config(model_name)
    rows = []

    for trial in range(1, trials + 1):
        config = deepcopy(base)
        config["training"].update(sample_training(rng, model_name))
        config["model"].update(sample_model(rng, model_name))
        config["training"]["epochs"] = epochs
        config["training"]["patience"] = epochs

        metrics = run_training(
            model_name,
            config,
            dashboard_port=None,
            save_artifacts=False,
            evaluate_test=False,
            verbose=False,
        )
        row = {
            **config["training"],
            **config["model"],
            "val_rank1": metrics["val"]["rank1"],
            "val_rank5": metrics["val"]["rank5"],
            "val_loss": metrics["val"]["loss"],
            "val_accuracy": metrics["val"]["accuracy"],
            "best_epoch": metrics["best_epoch"],
            "trial": trial,
        }
        rows.append(row)
        print(
            f"{model_name} trial {trial}/{trials} "
            f"val_rank1={row['val_rank1']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"lr={row['lr']:.2e} wd={row['weight_decay']:.2e} "
            f"batch={row['batch_size']} "
            f"dropout={row['dropout']:.3f}",
            flush=True,
        )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"tuning_{model_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = sorted(
        rows,
        key=lambda row: (row["val_rank1"], -row["val_loss"]),
        reverse=True,
    )[0]
    training_keys = [
        "lr",
        "weight_decay",
        "batch_size",
    ]
    model_keys = [
        key for key in base["model"] if key in best
    ]
    best_payload = {
        "training": {key: best[key] for key in training_keys},
        "model": {key: best[key] for key in model_keys},
        "val_rank1": best["val_rank1"],
        "val_loss": best["val_loss"],
        "trial": best["trial"],
    }
    (results_dir / f"best_{model_name}.json").write_text(
        json.dumps(best_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"{model_name}: best trial {best['trial']} "
        f"val_rank1={best['val_rank1']:.4f}",
        flush=True,
    )
    return best_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["resnet", "gru", "transformer", "all"],
        default="all",
    )
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_names = (
        ["resnet", "gru", "transformer"]
        if args.model == "all"
        else [args.model]
    )
    for model_name in model_names:
        tune_model(model_name, args.trials, args.epochs, args.seed)


if __name__ == "__main__":
    main()
