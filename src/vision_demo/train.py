from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from vision_demo.data import (
    class_names_from_records,
    discover_imagefolder,
    make_dataloaders,
    stratified_split,
    write_split_csv,
)
from vision_demo.metrics import classification_metrics
from vision_demo.model import SUPPORTED_ARCHITECTURES, create_model, freeze_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an explainable image classifier.")
    parser.add_argument("--data-dir", required=True, help="ImageFolder dataset root.")
    parser.add_argument("--output-dir", required=True, help="Run directory for outputs.")
    parser.add_argument("--arch", default="resnet18", choices=SUPPORTED_ARCHITECTURES)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels, _paths in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            total_loss += float(loss.item()) * images.size(0)

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.asarray(all_labels)
    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, y_true, y_prob


def save_training_plot(history: list[dict[str, float | int | None]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="accuracy")
    axes[1].plot(epochs, [row["val_macro_average_precision"] for row in history], label="mAP")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, discovered_class_names = discover_imagefolder(args.data_dir)
    split_records = stratified_split(
        records,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    class_names = class_names_from_records(split_records)
    if class_names != discovered_class_names:
        raise RuntimeError("Class label mapping changed during split creation.")

    write_split_csv(split_records, output_dir / "splits.csv")
    (output_dir / "classes.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    loaders = make_dataloaders(
        split_records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.arch, num_classes=len(class_names), pretrained=args.pretrained)
    if args.freeze_backbone:
        freeze_backbone(model, args.arch)
    model.to(device)

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float | int | None]] = []
    best_score = -1.0
    started = perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, _train_y, _train_prob = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
        )
        val_loss, val_y, val_prob = run_epoch(model, loaders["val"], criterion, device)
        val_metrics = classification_metrics(val_y, val_prob, class_names)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_auc_ovr": val_metrics["macro_auc_ovr"],
            "val_macro_average_precision": val_metrics["macro_average_precision"],
        }
        history.append(row)
        score = val_metrics["macro_average_precision"] or val_metrics["accuracy"] or 0.0
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "arch": args.arch,
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "pretrained": args.pretrained,
                    "freeze_backbone": args.freeze_backbone,
                },
                output_dir / "best_model.pt",
            )

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} val_map={val_metrics['macro_average_precision']}"
        )

    elapsed = perf_counter() - started
    (output_dir / "metrics_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_training_plot(history, output_dir / "training_curves.png")
    print(f"Training complete in {elapsed / 60:.1f} minutes. Best checkpoint: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
