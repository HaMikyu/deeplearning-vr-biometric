import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from configs import get_config
from datamodule import VRBiometricDataModule
from models import build_model

try:
    from dashboard import TrainingDashboard
except ImportError:
    class TrainingDashboard:
        def __init__(self, *args, **kwargs):
            self.enabled = False

        def update(self, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

        def stop(self):
            return None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    embeddings = []
    labels = []
    device_ids = []
    with torch.no_grad():
        for x, y, d in loader:
            x = x.to(device)
            y = y.to(device)
            logits, emb = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            embeddings.append(emb.detach().cpu())
            labels.append(y.detach().cpu())
            device_ids.append(d.detach().cpu())
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "embeddings": torch.cat(embeddings).numpy(),
        "labels": torch.cat(labels).numpy(),
        "device_ids": torch.cat(device_ids).numpy(),
    }


def ranking_metrics(query_embeddings, query_labels, gallery_embeddings, gallery_labels):
    query = F.normalize(torch.from_numpy(query_embeddings).float(), dim=1)
    gallery = F.normalize(torch.from_numpy(gallery_embeddings).float(), dim=1)
    similarities = query @ gallery.T
    gallery_labels_t = torch.from_numpy(gallery_labels)
    query_labels_t = torch.from_numpy(query_labels)
    top = similarities.topk(k=min(5, similarities.size(1)), dim=1).indices
    predicted = gallery_labels_t[top[:, 0]]
    rank1 = (predicted == query_labels_t).float().mean().item()
    top_labels = gallery_labels_t[top]
    rank5 = (top_labels == query_labels_t.unsqueeze(1)).any(dim=1).float().mean().item()
    return {"rank1": rank1, "rank5": rank5}


def save_embedding(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        embeddings=result["embeddings"],
        labels=result["labels"],
        device_ids=result["device_ids"],
    )


def run_training(
    model_name,
    config=None,
    dashboard_port=8000,
    save_artifacts=True,
    evaluate_test=True,
    verbose=True,
):
    if config is None:
        config = get_config(model_name)

    train_cfg = dict(config["training"])
    model_cfg = dict(config["model"])
    set_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = VRBiometricDataModule(
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        val_per_group=2,
        seed=train_cfg["seed"],
    )
    dm.setup()

    train_loader = DataLoader(
        dm.train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
    )
    eval_train_loader = DataLoader(
        dm.train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )
    val_loader = DataLoader(
        dm.val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )
    test_loader = DataLoader(
        dm.test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    model = build_model(
        model_name,
        n_features=dm.n_features,
        n_classes=dm.n_classes,
        **model_cfg,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=train_cfg.get("label_smoothing", 0.0)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, train_cfg["epochs"]),
    )

    config_text = (
        f"lr={train_cfg['lr']:.0e} "
        f"wd={train_cfg['weight_decay']:.0e} "
        f"batch={train_cfg['batch_size']} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    dashboard = TrainingDashboard(
        model_name=model_name,
        total_epochs=train_cfg["epochs"],
        config_text=config_text,
        port=dashboard_port,
    )

    history = []
    best_score = None
    best_state = None
    best_epoch = 0
    best_val_loss = None
    best_val_rank1 = None
    patience_counter = 0
    started = time.time()

    if verbose:
        print(
            f"{model_name}: device={device} "
            f"train={len(dm.train_ds)} val={len(dm.val_ds)} "
            f"test={len(dm.test_ds)} {config_text}"
        )

    try:
        for epoch in range(1, train_cfg["epochs"] + 1):
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for x, y, _ in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x)
                loss = criterion(logits, y)
                loss.backward()
                if train_cfg.get("grad_clip"):
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_cfg["grad_clip"],
                    )
                optimizer.step()
                train_loss_sum += loss.item() * x.size(0)
                train_total += x.size(0)
                train_correct += (logits.argmax(dim=1) == y).sum().item()

            train_loss = train_loss_sum / max(train_total, 1)
            train_accuracy = train_correct / max(train_total, 1)

            train_eval = evaluate(model, eval_train_loader, criterion, device)
            val_result = evaluate(model, val_loader, criterion, device)
            val_ranking = ranking_metrics(
                val_result["embeddings"],
                val_result["labels"],
                train_eval["embeddings"],
                train_eval["labels"],
            )

            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_result["loss"],
                "val_accuracy": val_result["accuracy"],
                "val_rank1": val_ranking["rank1"],
                "val_rank5": val_ranking["rank5"],
                "lr": lr,
            }
            history.append(entry)

            score = (val_ranking["rank1"], -val_result["loss"])
            improved = best_score is None or score > best_score
            if improved:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_epoch = epoch
                best_val_loss = val_result["loss"]
                best_val_rank1 = val_ranking["rank1"]
                patience_counter = 0
            else:
                patience_counter += 1

            if dashboard.enabled:
                dashboard.update(
                    status="running",
                    epoch=epoch,
                    total_epochs=train_cfg["epochs"],
                    train_loss=train_loss,
                    val_loss=val_result["loss"],
                    train_accuracy=train_accuracy,
                    val_accuracy=val_result["accuracy"],
                    val_rank1=val_ranking["rank1"],
                    best_rank1=best_val_rank1,
                    lr=lr,
                    history=history,
                    message=f"best epoch {best_epoch}, val Rank-1 {best_val_rank1:.4f}",
                )

            if verbose:
                print(
                    f"{model_name} {epoch:03d}/{train_cfg['epochs']:03d} "
                    f"train_loss={train_loss:.4f} "
                    f"train_acc={train_accuracy:.4f} "
                    f"val_loss={val_result['loss']:.4f} "
                    f"val_rank1={val_ranking['rank1']:.4f}"
                )

            if patience_counter >= train_cfg["patience"]:
                if verbose:
                    print(f"{model_name}: early stop na epoce {epoch}")
                break

        model.load_state_dict(best_state)
        train_result = evaluate(model, eval_train_loader, criterion, device)
        val_result = evaluate(model, val_loader, criterion, device)
        val_ranking = ranking_metrics(
            val_result["embeddings"],
            val_result["labels"],
            train_result["embeddings"],
            train_result["labels"],
        )

        test_result = None
        test_ranking = None
        if evaluate_test:
            test_result = evaluate(model, test_loader, criterion, device)
            test_ranking = ranking_metrics(
                test_result["embeddings"],
                test_result["labels"],
                train_result["embeddings"],
                train_result["labels"],
            )

        metrics = {
            "model": model_name,
            "config": config,
            "parameters": sum(p.numel() for p in model.parameters()),
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "train": {
                "loss": train_result["loss"],
                "accuracy": train_result["accuracy"],
            },
            "val": {
                "loss": val_result["loss"],
                "accuracy": val_result["accuracy"],
                **val_ranking,
            },
            "test": (
                {
                    "loss": test_result["loss"],
                    "accuracy": test_result["accuracy"],
                    **test_ranking,
                }
                if test_result is not None
                else None
            ),
            "history": history,
        }

        if save_artifacts:
            save_embedding(
                Path("embeddings") / model_name / "train_embedding.npz",
                train_result,
            )
            save_embedding(
                Path("embeddings") / model_name / "val_embedding.npz",
                val_result,
            )
            if test_result is not None:
                save_embedding(
                    Path("embeddings") / model_name / "test_embedding.npz",
                    test_result,
                )

            checkpoint_path = (
                Path("models") / "checkpoints" / f"{model_name}.pt"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": best_state,
                    "config": config,
                    "n_features": dm.n_features,
                    "n_classes": dm.n_classes,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )

            results_dir = Path("results")
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / f"{model_name}_metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            with (results_dir / f"{model_name}_history.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)

        if dashboard.enabled:
            dashboard.update(
                status="finished",
                epoch=len(history),
                total_epochs=train_cfg["epochs"],
                val_loss=val_result["loss"],
                val_accuracy=val_result["accuracy"],
                val_rank1=val_ranking["rank1"],
                best_rank1=best_val_rank1,
                history=history,
                message=(
                    f"best epoch {best_epoch}, "
                    f"test Rank-1 {metrics['test']['rank1']:.4f}"
                    if metrics["test"] is not None
                    else f"best epoch {best_epoch}"
                ),
            )
            dashboard.finish(
                "finished",
                f"Czas: {time.time() - started:.1f}s, best epoch {best_epoch}",
            )

        if verbose:
            test_text = (
                f"test_rank1={metrics['test']['rank1']:.4f}"
                if metrics["test"] is not None
                else "test=off"
            )
            print(
                f"{model_name}: best_epoch={best_epoch} "
                f"val_rank1={val_ranking['rank1']:.4f} {test_text}"
            )

        return metrics
    except Exception as error:
        if dashboard.enabled:
            dashboard.finish("failed", str(error))
        raise
    finally:
        if dashboard.enabled:
            dashboard.stop()


def cli(model_name):
    config = get_config(model_name)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=config["training"]["epochs"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config["training"]["batch_size"],
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config["training"]["lr"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config["training"]["seed"],
    )
    parser.add_argument("--dashboard-port", type=int, default=8000)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args()

    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["lr"] = args.lr
    config["training"]["seed"] = args.seed

    run_training(
        model_name,
        config,
        dashboard_port=None if args.no_dashboard else args.dashboard_port,
        evaluate_test=not args.no_test,
    )
