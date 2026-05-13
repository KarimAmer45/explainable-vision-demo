import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from xai_vision_demo.explain import GradCAM, overlay_heatmap


class TinyCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.classifier = nn.Linear(4, 2)

    def forward(self, image_tensor):
        features = F.relu(self.conv(image_tensor))
        pooled = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        return self.classifier(pooled)


def test_overlay_heatmap_returns_rgb_image_with_input_size():
    image = Image.new("RGB", (8, 8), "white")
    heatmap = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)

    overlay = overlay_heatmap(image, heatmap)

    assert overlay.mode == "RGB"
    assert overlay.size == image.size


def test_gradcam_returns_normalized_heatmap_for_target_layer():
    torch.manual_seed(7)
    model = TinyCamModel()
    model.eval()
    image_tensor = torch.randn(1, 3, 12, 12, requires_grad=True)
    gradcam = GradCAM(model, model.conv)

    try:
        heatmap = gradcam(image_tensor, class_index=1)
    finally:
        gradcam.close()

    assert heatmap.shape == (12, 12)
    assert np.isfinite(heatmap).all()
    assert 0.0 <= float(heatmap.min()) <= 1.0
    assert 0.0 <= float(heatmap.max()) <= 1.0
