from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, float | None]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    labels = np.arange(len(class_names))
    y_true_binary = label_binarize(y_true, classes=labels)

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_auc_ovr": None,
        "macro_average_precision": None,
    }
    if len(class_names) == 2:
        try:
            metrics["macro_auc_ovr"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except ValueError:
            metrics["macro_auc_ovr"] = None

        try:
            metrics["macro_average_precision"] = float(average_precision_score(y_true, y_prob[:, 1]))
        except ValueError:
            metrics["macro_average_precision"] = None
        return metrics

    try:
        metrics["macro_auc_ovr"] = float(
            roc_auc_score(y_true_binary, y_prob, average="macro", multi_class="ovr")
        )
    except ValueError:
        metrics["macro_auc_ovr"] = None

    try:
        metrics["macro_average_precision"] = float(
            average_precision_score(y_true_binary, y_prob, average="macro")
        )
    except ValueError:
        metrics["macro_average_precision"] = None

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    y_pred = np.asarray(y_prob).argmax(axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, matrix[row, col], ha="center", va="center", color=color)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    labels = np.arange(len(class_names))
    y_true_binary = label_binarize(y_true, classes=labels)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    for index, class_name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_true_binary[:, index], y_prob[:, index])
        except ValueError:
            continue
        ax.plot(fpr, tpr, label=class_name)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#667085", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
