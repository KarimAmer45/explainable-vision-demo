from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    image_path: str
    class_name: str
    label: int
    split: str | None = None


def discover_imagefolder(data_dir: str | Path) -> tuple[list[ImageRecord], list[str]]:
    root = Path(data_dir)
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found in {root}")

    class_names = [path.name for path in class_dirs]
    records: list[ImageRecord] = []
    for label, class_dir in enumerate(class_dirs):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    ImageRecord(
                        image_path=str(image_path),
                        class_name=class_dir.name,
                        label=label,
                    )
                )

    if not records:
        raise ValueError(f"No images found under class folders in {root}")
    return records, class_names


def stratified_split(
    records: list[ImageRecord],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> list[ImageRecord]:
    labels = [record.label for record in records]
    holdout_size = val_size + test_size
    if not 0 < holdout_size < 1:
        raise ValueError("val_size + test_size must be between 0 and 1")

    train_records, holdout_records = train_test_split(
        records,
        test_size=holdout_size,
        stratify=labels,
        random_state=seed,
    )
    holdout_labels = [record.label for record in holdout_records]
    relative_test_size = test_size / holdout_size
    val_records, test_records = train_test_split(
        holdout_records,
        test_size=relative_test_size,
        stratify=holdout_labels,
        random_state=seed,
    )

    split_records: list[ImageRecord] = []
    for split_name, split_items in (
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ):
        split_records.extend(
            ImageRecord(
                image_path=record.image_path,
                class_name=record.class_name,
                label=record.label,
                split=split_name,
            )
            for record in split_items
        )
    return sorted(split_records, key=lambda record: (record.split or "", record.image_path))


def write_split_csv(records: list[ImageRecord], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "class_name", "label", "split"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": record.image_path,
                    "class_name": record.class_name,
                    "label": record.label,
                    "split": record.split,
                }
            )


def read_split_csv(split_csv: str | Path) -> list[ImageRecord]:
    with Path(split_csv).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            ImageRecord(
                image_path=row["image_path"],
                class_name=row["class_name"],
                label=int(row["label"]),
                split=row["split"],
            )
            for row in reader
        ]


def class_names_from_records(records: list[ImageRecord]) -> list[str]:
    by_label = {record.label: record.class_name for record in records}
    return [by_label[label] for label in sorted(by_label)]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class SplitImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: transforms.Compose):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        return self.transform(image), record.label, record.image_path


def make_dataloaders(
    records: list[ImageRecord],
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split_name in ("train", "val", "test"):
        split_records = [record for record in records if record.split == split_name]
        if not split_records:
            continue
        dataset = SplitImageDataset(
            split_records,
            transform=build_transforms(image_size=image_size, train=split_name == "train"),
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
