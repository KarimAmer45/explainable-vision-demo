from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from xai_vision_demo.data import build_transforms
from xai_vision_demo.explain import overlay_heatmap
from xai_vision_demo.metrics import classification_metrics
from xai_vision_demo.model import create_model, freeze_backbone, is_vit_arch
from xai_vision_demo.train import run_epoch, save_training_plot, set_seed
from xai_vision_demo.transformer_explain import attention_rollout_heatmap


class DatasetWithIds(Dataset):
    def __init__(self, dataset: Dataset, prefix: str):
        self.dataset = dataset
        self.prefix = prefix

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image, label = self.dataset[index]
        return image, int(label), f"{self.prefix}:{index}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a ViT/Swin classifier on CIFAR-10 and export transformer XAI assets."
    )
    parser.add_argument("--output-dir", default="runs/cifar10_vit_b_16")
    parser.add_argument("--data-dir", default="data/public")
    parser.add_argument("--arch", choices=["vit_b_16", "swin_t"], default="vit_b_16")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.12)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-val-samples", type=int, default=500)
    parser.add_argument("--max-test-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention-discard-ratio", type=float, default=0.0)
    return parser.parse_args()


def stratified_limit(
    indices: np.ndarray,
    labels: np.ndarray,
    max_items: int | None,
    seed: int,
) -> list[int]:
    if max_items is None or max_items >= len(indices):
        return sorted(int(index) for index in indices)
    if max_items < len(np.unique(labels[indices])):
        raise ValueError("Sample limits must be at least the number of classes.")

    _unused, limited = train_test_split(
        indices,
        test_size=max_items,
        stratify=labels[indices],
        random_state=seed,
    )
    return sorted(int(index) for index in limited)


def make_cifar10_loaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    val_size: float,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    seed: int,
    num_workers: int,
    download: bool,
) -> tuple[dict[str, DataLoader], list[str], dict[str, list[int]]]:
    train_raw = datasets.CIFAR10(root=data_dir, train=True, download=download)
    test_raw = datasets.CIFAR10(root=data_dir, train=False, download=download)
    class_names = list(train_raw.classes)

    train_labels = np.asarray(train_raw.targets)
    test_labels = np.asarray(test_raw.targets)
    train_indices, val_indices = train_test_split(
        np.arange(len(train_labels)),
        test_size=val_size,
        stratify=train_labels,
        random_state=seed,
    )
    train_indices = stratified_limit(train_indices, train_labels, max_train_samples, seed)
    val_indices = stratified_limit(val_indices, train_labels, max_val_samples, seed)
    test_indices = stratified_limit(np.arange(len(test_labels)), test_labels, max_test_samples, seed)

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=build_transforms(image_size=image_size, train=True),
    )
    eval_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=build_transforms(image_size=image_size, train=False),
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=build_transforms(image_size=image_size, train=False),
    )

    loaders = {
        "train": DataLoader(
            DatasetWithIds(Subset(train_dataset, train_indices), "cifar10-train"),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            DatasetWithIds(Subset(eval_train_dataset, val_indices), "cifar10-val"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            DatasetWithIds(Subset(test_dataset, test_indices), "cifar10-test"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    return loaders, class_names, {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def save_attention_rollout_example(
    model: nn.Module,
    data_dir: Path,
    output_path: Path,
    image_size: int,
    test_indices: list[int],
    device: torch.device,
    discard_ratio: float,
) -> None:
    if not test_indices:
        return

    raw_test = datasets.CIFAR10(root=data_dir, train=False, download=False)
    image, _label = raw_test[test_indices[0]]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    transform = build_transforms(image_size=image_size, train=False)
    image_tensor = transform(image).unsqueeze(0).to(device)
    heatmap = attention_rollout_heatmap(model, image_tensor, discard_ratio=discard_ratio)
    overlay = overlay_heatmap(image.resize((image_size, image_size)), heatmap)
    overlay.save(output_path)


def save_experiment_card(
    output_dir: Path,
    args: argparse.Namespace,
    test_metrics: dict[str, float | None],
    split_indices: dict[str, list[int]],
) -> None:
    card = f"""# CIFAR-10 Transformer Experiment

## Setup

- Dataset: CIFAR-10
- Architecture: `{args.arch}`
- Pretrained weights: `{args.pretrained}`
- Frozen backbone: `{args.freeze_backbone}`
- Image size: `{args.image_size}`
- Train / validation / test samples: `{len(split_indices["train"])}` / `{len(split_indices["val"])}` / `{len(split_indices["test"])}`

## Test Metrics

- Accuracy: `{test_metrics["accuracy"]:.3f}`
- Macro ROC-AUC OVR: `{test_metrics["macro_auc_ovr"]:.3f}`
- Macro average precision: `{test_metrics["macro_average_precision"]:.3f}`
- Loss: `{test_metrics["loss"]:.3f}`

## Explainability

ViT checkpoints export `attention_rollout_example.png`, which averages attention heads,
adds the residual identity path, rolls attention through the transformer stack, and
projects class-token attention back to image space.

## Artifacts

- `best_model.pt`
- `metrics_history.json`
- `test_metrics.json`
- `training_curves.png`
- `attention_rollout_example.png` for ViT runs
"""
    (output_dir / "experiment_card.md").write_text(card, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    loaders, class_names, split_indices = make_cifar10_loaders(
        data_dir=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_size=args.val_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
        num_workers=args.num_workers,
        download=args.download,
    )
    (output_dir / "classes.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "dataset_indices.json").write_text(
        json.dumps(split_indices, indent=2),
        encoding="utf-8",
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
                    "dataset": "CIFAR10",
                    "split_indices": split_indices,
                },
                output_dir / "best_model.pt",
            )

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} val_map={val_metrics['macro_average_precision']}"
        )

    test_loss, test_y, test_prob = run_epoch(model, loaders["test"], criterion, device)
    test_metrics = classification_metrics(test_y, test_prob, class_names)
    test_metrics["loss"] = test_loss

    (output_dir / "metrics_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    save_experiment_card(output_dir, args, test_metrics, split_indices)
    save_training_plot(history, output_dir / "training_curves.png")

    if is_vit_arch(args.arch):
        save_attention_rollout_example(
            model=model,
            data_dir=data_dir,
            output_path=output_dir / "attention_rollout_example.png",
            image_size=args.image_size,
            test_indices=split_indices["test"],
            device=device,
            discard_ratio=args.attention_discard_ratio,
        )

    elapsed = perf_counter() - started
    print(json.dumps(test_metrics, indent=2))
    print(f"Transformer experiment complete in {elapsed / 60:.1f} minutes: {output_dir}")


if __name__ == "__main__":
    main()
