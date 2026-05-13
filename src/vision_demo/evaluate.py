from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from vision_demo.data import make_dataloaders, read_split_csv
from vision_demo.metrics import classification_metrics, plot_confusion_matrix, plot_roc_curves
from vision_demo.model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained image classifier.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint["class_names"]
    arch = checkpoint["arch"]
    image_size = checkpoint["image_size"]

    model = create_model(arch, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    records = read_split_csv(args.split_csv)
    loaders = make_dataloaders(
        records,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = loaders[args.split]
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []
    all_paths: list[str] = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc=f"evaluate:{args.split}"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.asarray(all_labels)
    metrics = classification_metrics(y_true, y_prob, class_names)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["image_path", "label", "prediction", "confidence", *class_names]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image_path, label, probs in zip(all_paths, y_true, y_prob, strict=True):
            prediction = int(np.argmax(probs))
            row = {
                "image_path": image_path,
                "label": class_names[int(label)],
                "prediction": class_names[prediction],
                "confidence": float(probs[prediction]),
            }
            row.update({class_name: float(probs[index]) for index, class_name in enumerate(class_names)})
            writer.writerow(row)

    plot_confusion_matrix(y_true, y_prob, class_names, output_dir / "confusion_matrix.png")
    plot_roc_curves(y_true, y_prob, class_names, output_dir / "roc_curves.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
