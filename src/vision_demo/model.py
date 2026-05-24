from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHITECTURES = ("resnet18", "efficientnet_b0", "vit_b_16", "swin_t")


def create_model(
    arch: str,
    num_classes: int,
    pretrained: bool = False,
) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model

    if arch == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture: {arch}")


def freeze_backbone(model: nn.Module, arch: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    arch = arch.lower()
    if arch == "resnet18":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    if arch == "efficientnet_b0":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    if arch == "vit_b_16":
        for parameter in model.heads.parameters():
            parameter.requires_grad = True
        return

    if arch == "swin_t":
        for parameter in model.head.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(f"Unsupported architecture: {arch}")


def is_vit_arch(arch: str) -> bool:
    return arch.lower() == "vit_b_16"


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        return model.layer4[-1]
    if arch == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"GradCAM target layer is not defined for architecture: {arch}")
