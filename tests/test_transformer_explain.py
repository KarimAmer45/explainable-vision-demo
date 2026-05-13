import numpy as np
import torch
from torchvision.models.vision_transformer import VisionTransformer

from vision_demo.transformer_explain import attention_rollout_heatmap


def test_attention_rollout_returns_normalized_image_sized_heatmap():
    torch.manual_seed(7)
    model = VisionTransformer(
        image_size=16,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
        num_classes=3,
    )
    model.eval()
    image_tensor = torch.randn(1, 3, 16, 16)

    heatmap = attention_rollout_heatmap(model, image_tensor)

    assert heatmap.shape == (16, 16)
    assert np.isfinite(heatmap).all()
    assert 0.0 <= float(heatmap.min()) <= 1.0
    assert 0.0 <= float(heatmap.max()) <= 1.0
