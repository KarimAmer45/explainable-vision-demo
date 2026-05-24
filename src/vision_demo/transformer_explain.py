from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import MethodType

import numpy as np
import torch


@contextmanager
def _capture_vit_attention(model: torch.nn.Module) -> Iterator[list[torch.Tensor]]:
    """Force torchvision ViT attention blocks to return per-head attention weights."""
    attention_maps: list[torch.Tensor] = []
    patched_modules: list[tuple[torch.nn.Module, object]] = []

    for layer in model.encoder.layers:
        attention = layer.self_attention
        original_forward = attention.forward

        def forward_with_weights(self, *args, _original_forward=original_forward, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            output, weights = _original_forward(*args, **kwargs)
            attention_maps.append(weights.detach())
            return output, weights

        attention.forward = MethodType(forward_with_weights, attention)
        patched_modules.append((attention, original_forward))

    try:
        yield attention_maps
    finally:
        for attention, original_forward in patched_modules:
            attention.forward = original_forward


def attention_rollout_heatmap(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """Return a normalized ViT class-token attention rollout heatmap.

    The implementation follows the common rollout recipe: average per-head attention,
    add residual identity, row-normalize, multiply attention matrices across blocks,
    and reshape the class-token attention over patch tokens into a spatial map.
    """
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError("attention_rollout_heatmap expects a single image tensor shaped [1, C, H, W].")
    if not 0.0 <= discard_ratio < 1.0:
        raise ValueError("discard_ratio must be in the range [0, 1).")
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        raise ValueError("attention rollout requires a torchvision VisionTransformer model.")

    was_training = model.training
    model.eval()
    with torch.no_grad(), _capture_vit_attention(model) as attention_maps:
        model(image_tensor)

    if was_training:
        model.train()

    if not attention_maps:
        raise RuntimeError("No ViT attention maps were captured.")

    rollout = torch.eye(attention_maps[0].shape[-1], device=image_tensor.device).unsqueeze(0)
    for attention in attention_maps:
        attention = attention.to(image_tensor.device)
        attention = attention.mean(dim=1)
        if discard_ratio > 0:
            flat = attention[:, 1:, 1:].reshape(attention.shape[0], -1)
            cutoff = int(flat.shape[1] * discard_ratio)
            if cutoff > 0:
                threshold = flat.kthvalue(cutoff, dim=1).values.reshape(-1, 1, 1)
                attention[:, 1:, 1:] = torch.where(
                    attention[:, 1:, 1:] < threshold,
                    torch.zeros_like(attention[:, 1:, 1:]),
                    attention[:, 1:, 1:],
                )
        attention = attention + torch.eye(attention.shape[-1], device=attention.device).unsqueeze(0)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        rollout = attention.bmm(rollout)

    class_attention = rollout[0, 0, 1:]
    grid_size = int(class_attention.numel() ** 0.5)
    if grid_size * grid_size != class_attention.numel():
        raise RuntimeError("ViT patch tokens do not form a square attention grid.")

    heatmap = class_attention.reshape(grid_size, grid_size)
    heatmap = torch.nn.functional.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    heatmap = heatmap.cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap
