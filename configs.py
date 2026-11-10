import json
from copy import deepcopy
from pathlib import Path

COMMON = {
    "epochs": 60,
    "patience": 12,
    "batch_size": 32,
    "seed": 42,
    "label_smoothing": 0.05,
    "num_workers": 0,
    "grad_clip": 1.0,
}

MODEL_CONFIGS = {
    "resnet": {
        "training": {
            **COMMON,
            "lr": 1e-3,
            "weight_decay": 1e-2,
        },
        "model": {
            "embedding_dim": 128,
            "dropout": 0.1,
            "base_channels": 64,
        },
    },
    "gru": {
        "training": {
            **COMMON,
            "lr": 1e-3,
            "weight_decay": 1e-2,
        },
        "model": {
            "embedding_dim": 128,
            "dropout": 0.2,
            "hidden_size": 96,
            "num_layers": 2,
        },
    },
    "transformer": {
        "training": {
            **COMMON,
            "lr": 5e-4,
            "weight_decay": 1e-2,
        },
        "model": {
            "embedding_dim": 128,
            "dropout": 0.1,
            "d_model": 96,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 192,
        },
    },
}


def get_config(model_name):
    config = deepcopy(MODEL_CONFIGS[model_name])
    best_path = Path("results") / f"best_{model_name}.json"
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        config["training"].update(best.get("training", {}))
        config["model"].update(best.get("model", {}))
    return config
