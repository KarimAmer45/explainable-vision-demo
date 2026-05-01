from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from xai_vision_demo.data import build_transforms
from xai_vision_demo.model import create_model, gradcam_target_layer


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image_tensor: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        score = logits[:, class_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    image_array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    cmap = plt.get_cmap("magma")
    colored_heatmap = cmap(heatmap)[..., :3]
    blended = (1 - alpha) * image_array + alpha * colored_heatmap
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(
        checkpoint["arch"],
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def explain_image(
    checkpoint_path: str | Path,
    image_path: str | Path,
    class_index: int | None = None,
    output_path: str | Path | None = None,
) -> tuple[Image.Image, dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    transform = build_transforms(checkpoint["image_size"], train=False)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(image_tensor), dim=1).squeeze(0).cpu().numpy()
    target_layer = gradcam_target_layer(model, checkpoint["arch"])
    gradcam = GradCAM(model, target_layer)
    try:
        heatmap = gradcam(image_tensor, class_index=class_index)
    finally:
        gradcam.close()

    resized_image = image.resize((checkpoint["image_size"], checkpoint["image_size"]))
    overlay = overlay_heatmap(resized_image, heatmap)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(output_path)

    scores = {
        class_name: float(probs[index]) for index, class_name in enumerate(checkpoint["class_names"])
    }
    return overlay, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a GradCAM explanation for one image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-index", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overlay, scores = explain_image(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        class_index=args.class_index,
        output_path=args.output,
    )
    top_class, top_score = max(scores.items(), key=lambda item: item[1])
    print(f"Saved {args.output}. Top class: {top_class} ({top_score:.3f})")


if __name__ == "__main__":
    main()
