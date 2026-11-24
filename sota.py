"""SOTA: ArcFace + augmentacje + EMA + cechy odporne na zmiane dnia + ensemble.

Protokol jak w baseline: trening na dniu 1, galeria = probki treningowe dnia 1,
probe = dzien 2, podobienstwo cosinusowe embeddingow.

Zeby nie dobierac przepisu pod wynik testowy, dzien 2 jest dzielony po rzutach:
- dev  = rzuty 0-4 kazdej osoby i urzadzenia (wybor przepisu, `screen`),
- test = rzuty 5-9 (nietkniete do konca, raportowane w `final`).
Dla porownania z baseline raportujemy tez caly dzien 2 (`all`).

Uzycie:
    python sota.py screen --models transformer resnet --variants base arcface full
    python sota.py final --models transformer resnet inception gru
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from configs import get_config
from datamodule import VRBiometricDataModule
from metrics import evaluate as embedding_metrics
from models import build_model
from train_common import TensorBatches, save_embedding, set_seed

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

RESULTS = Path("results")

# Przepis domyslny = pelny SOTA. Warianty w VARIANTS nadpisuja pojedyncze pola.
RECIPE = {
    "epochs": 150,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 5e-2,
    "warmup": 5,
    "grad_clip": 1.0,
    "loss": "arcface",  # "arcface" albo "ce"
    "arc_s": 30.0,
    "arc_m": 0.3,
    "label_smoothing": 0.0,
    "linear_embedding": True,
    "augment": 1.0,  # sila augmentacji, 0 = wylaczone
    "ema": 0.995,  # decay EMA wag na krok, 0 = wylaczone
    "center_xz": True,
    "velocity": True,
    "full_day1": False,  # trening na calym dniu 1 (bez podzialu val)
}

BASELINE = {
    "loss": "ce",
    "label_smoothing": 0.05,
    "linear_embedding": False,
    "augment": 0.0,
    "ema": 0.0,
    "center_xz": False,
    "velocity": False,
}

VARIANTS = {
    # kolejne kroki od baseline do pelnego przepisu (ablacja skladnikow)
    "base": {**BASELINE},
    "arcface": {**BASELINE, "loss": "arcface", "label_smoothing": 0.0, "linear_embedding": True},
    "arc_aug": {**BASELINE, "loss": "arcface", "label_smoothing": 0.0, "linear_embedding": True,
                "augment": 1.0},
    "arc_aug_ema": {**BASELINE, "loss": "arcface", "label_smoothing": 0.0, "linear_embedding": True,
                    "augment": 1.0, "ema": 0.995},
    "arc_aug_ema_center": {"velocity": False},
    "full": {},
    # pojedyncze zmiany wzgledem pelnego przepisu
    "full_ce": {"loss": "ce", "label_smoothing": 0.05, "linear_embedding": False},
    "full_noaug": {"augment": 0.0},
    "full_nocenter": {"center_xz": False},
    "full_aug2": {"augment": 2.0},
    "full_m02": {"arc_m": 0.2},
    "full_m05": {"arc_m": 0.5},
    "full_s64": {"arc_s": 64.0},
    "full_e100": {"epochs": 100},
    "full_e250": {"epochs": 250},
    "full_wd1e2": {"weight_decay": 1e-2},
    "full_lr2e3": {"lr": 2e-3},
    "full_bs32": {"batch_size": 32},
    "full_day1": {"full_day1": True},
}

INCEPTION_MODEL = {"embedding_dim": 128, "n_filters": 32, "depth": 6, "dropout": 0.1}


def model_config(model_name):
    if model_name == "inception":
        return dict(INCEPTION_MODEL)
    # hiperparametry architektury ze strojenia baseline (results/best_<model>.json)
    return dict(get_config(model_name)["model"])


class ArcFace(nn.Module):
    """Additive angular margin (Deng i in. 2019): logit celu = s*cos(theta + m)."""

    def __init__(self, embedding_dim, n_classes, s=30.0, m=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, y):
        cos = F.normalize(emb) @ F.normalize(self.weight).T
        sin = (1.0 - cos.pow(2)).clamp(min=1e-7).sqrt()
        phi = cos * math.cos(self.m) - sin * math.sin(self.m)
        # gdy theta + m > pi, cos(theta + m) przestaje malec: liniowa kara
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        onehot = F.one_hot(y, cos.size(1)).bool()
        return self.s * torch.where(onehot, phi, cos)


def augment(x, strength=1.0):
    """Augmentacje na GPU dla batcha (B, T, C) w jednostkach znormalizowanych.

    - losowe przyciecie i rozciagniecie w czasie (zmiana tempa rzutu),
    - skalowanie amplitudy kanalow,
    - szum gaussowski.
    """
    if strength <= 0:
        return x
    b, t, c = x.shape
    dev = x.device
    scale = 1.0 - 0.15 * strength * torch.rand(b, 1, device=dev)
    shift = (torch.rand(b, 1, device=dev) * 2 - 1) * (1 - scale)
    grid_x = torch.linspace(-1, 1, t, device=dev)[None] * scale + shift
    grid = torch.stack([grid_x, torch.zeros_like(grid_x)], dim=-1).unsqueeze(1)
    x = F.grid_sample(
        x.transpose(1, 2).unsqueeze(2), grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    ).squeeze(2).transpose(1, 2)
    x = x * (1 + 0.05 * strength * torch.randn(b, 1, c, device=dev))
    return x + 0.03 * strength * torch.randn_like(x)


@torch.no_grad()
def embed(model, loader):
    model.eval()
    out = [model(x)[1].float().cpu() for x, _, _ in loader]
    return torch.cat(out).numpy()


def split_metrics(test_emb, test_y, test_d, throw, gal_emb, gal_y):
    out = {}
    for name, mask in (("dev", throw < 5), ("test", throw >= 5), ("all", throw >= 0)):
        m = embedding_metrics(test_emb[mask], test_y[mask], test_d[mask], gal_emb, gal_y)
        out.update({f"{name}_{k}": v for k, v in m.items()})
    return out


def run(model_name, recipe, seed, modality="full"):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = VRBiometricDataModule(
        modality=modality, val_per_group=2, seed=seed,
        center_xz=recipe["center_xz"], velocity=recipe["velocity"],
    )
    dm.setup()
    base = dm.train_ds.dataset
    train_idx = np.arange(len(base)) if recipe["full_day1"] else dm.train_idx
    train = TensorBatches(base, train_idx, recipe["batch_size"], device, shuffle=True, seed=seed)
    gallery = TensorBatches(base, train_idx, 512, device)
    val = TensorBatches(base, dm.val_idx, 512, device)
    test = TensorBatches(dm.test_ds, None, 512, device)

    cfg = {**model_config(model_name), "linear_embedding": recipe["linear_embedding"]}
    model = build_model(model_name, n_features=dm.n_features, n_classes=dm.n_classes, **cfg).to(device)
    head = None
    params = list(model.parameters())
    if recipe["loss"] == "arcface":
        head = ArcFace(cfg["embedding_dim"], dm.n_classes, recipe["arc_s"], recipe["arc_m"]).to(device)
        params += list(head.parameters())

    opt = torch.optim.AdamW(params, lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    total = recipe["epochs"] * len(train)
    warm = max(1, recipe["warmup"] * len(train))

    def lr_lambda(step):
        return min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = (
        AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(recipe["ema"]), use_buffers=True)
        if recipe["ema"] > 0 else None
    )

    started = time.time()
    for _ in range(recipe["epochs"]):
        model.train()
        if head is not None:
            head.train()
        for x, y, _ in train:
            x = augment(x, recipe["augment"])
            logits, emb = model(x)
            if head is not None:
                logits = head(emb, y)
            loss = F.cross_entropy(logits, y, label_smoothing=recipe["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, recipe["grad_clip"])
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)

    final = ema.module if ema is not None else model
    gal_emb, val_emb, test_emb = embed(final, gallery), embed(final, val), embed(final, test)
    gal_y = gallery.y.cpu().numpy()
    val_y, val_d = val.y.cpu().numpy(), val.d.cpu().numpy()
    test_y, test_d = test.y.cpu().numpy(), test.d.cpu().numpy()

    metrics = split_metrics(test_emb, test_y, test_d, dm.test_throw, gal_emb, gal_y)
    metrics["val_rank1"] = embedding_metrics(val_emb, val_y, val_d, gal_emb, gal_y)["rank1"]
    metrics["seconds"] = time.time() - started
    metrics["parameters"] = sum(p.numel() for p in model.parameters())
    embeddings = {
        "train": {"embeddings": gal_emb, "labels": gal_y, "device_ids": gallery.d.cpu().numpy()},
        "val": {"embeddings": val_emb, "labels": val_y, "device_ids": val_d},
        "test": {"embeddings": test_emb, "labels": test_y, "device_ids": test_d},
        "test_throw": dm.test_throw,
    }
    return metrics, embeddings


def ensemble(emb_list):
    """Konkatenacja embeddingow znormalizowanych L2 (= srednia podobienstw cosinusowych)."""
    def cat(split):
        parts = [e[split]["embeddings"] for e in emb_list]
        parts = [p / np.linalg.norm(p, axis=1, keepdims=True).clip(1e-12) for p in parts]
        return np.concatenate(parts, axis=1) / math.sqrt(len(parts))

    first = emb_list[0]
    for e in emb_list[1:]:
        assert np.array_equal(e["train"]["labels"], first["train"]["labels"])
    return split_metrics(
        cat("test"), first["test"]["labels"], first["test"]["device_ids"],
        first["test_throw"], cat("train"), first["train"]["labels"],
    ), {
        split: {**first[split], "embeddings": cat(split)} for split in ("train", "val", "test")
    }


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def fmt(m, split):
    return (
        f"{split}: rank1={m[f'{split}_rank1']:.4f} rank5={m[f'{split}_rank5']:.4f} "
        f"eer={m[f'{split}_eer']:.4f}"
    )


def screen(args):
    path = RESULTS / f"sota_screen_{'_'.join(args.models)}.csv"
    for variant in args.variants:
        recipe = {**RECIPE, **VARIANTS[variant]}
        for model_name in args.models:
            for seed in args.seeds:
                m, _ = run(model_name, recipe, seed)
                append_csv(path, {"variant": variant, "model": model_name, "seed": seed, **m})
                print(f"{variant:20s} {model_name:11s} seed={seed} {fmt(m, 'dev')} "
                      f"({m['seconds']:.0f}s)", flush=True)


def final(args):
    variant = args.variant
    recipe = {**RECIPE, **VARIANTS[variant]}
    rows = []
    for seed in args.seeds:
        per_model = {}
        for model_name in args.models:
            m, emb = run(model_name, recipe, seed)
            per_model[model_name] = emb
            rows.append({"variant": variant, "run": model_name, "seed": seed, **m})
            print(f"{model_name:24s} seed={seed} {fmt(m, 'test')} | {fmt(m, 'all')}", flush=True)
            if seed == args.seeds[0]:
                for split in ("train", "val", "test"):
                    save_embedding(Path("embeddings") / f"sota_{model_name}" / f"{split}_embedding.npz",
                                   emb[split])
        for combo in args.ensembles:
            names = combo.split("+")
            m, emb = ensemble([per_model[n] for n in names])
            rows.append({"variant": variant, "run": f"ensemble_{combo}", "seed": seed, **m})
            print(f"{'ens ' + combo:24s} seed={seed} {fmt(m, 'test')} | {fmt(m, 'all')}", flush=True)
            if seed == args.seeds[0]:
                for split in ("train", "val", "test"):
                    save_embedding(Path("embeddings") / f"sota_ensemble_{combo.replace('+', '_')}"
                                   / f"{split}_embedding.npz", emb[split])

    out = RESULTS / f"sota_final_{variant}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"variant": variant, "recipe": recipe, "seeds": args.seeds, "runs": {}}
    for run_name in dict.fromkeys(row["run"] for row in rows):
        group = [row for row in rows if row["run"] == run_name]
        stats = {}
        for key in group[0]:
            if key in ("variant", "run", "seed"):
                continue
            values = np.array([row[key] for row in group], dtype=float)
            stats[key] = {"mean": float(values.mean()),
                          "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
        summary["runs"][run_name] = stats
    (RESULTS / f"sota_summary_{variant}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"zapisano {out}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("screen")
    p.add_argument("--models", nargs="+", default=["transformer", "resnet"])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p = sub.add_parser("final")
    # przepis wybrany na dev (patrz README / raport, sekcja S4)
    p.add_argument("--variant", default="full_day1", choices=list(VARIANTS))
    p.add_argument("--models", nargs="+", default=["transformer", "resnet", "inception", "gru"])
    p.add_argument("--ensembles", nargs="*",
                   default=["transformer+resnet+inception+gru", "transformer+resnet+inception"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    screen(args) if args.cmd == "screen" else final(args)


if __name__ == "__main__":
    main()
