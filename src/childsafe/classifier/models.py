"""Model factory (timm) and checkpoint I/O."""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn

ARCHS = ("efficientnet_b0", "mobilenetv3_large_100", "vgg19_bn")


def create_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Last spatial feature layer. (In timm MobileNetV3 conv_head comes after pooling, so use blocks.)"""
    if arch.startswith(("efficientnet", "mobilenetv3")):
        return model.blocks[-1]
    if arch.startswith("vgg"):
        return model.pre_logits
    raise ValueError(f"no Grad-CAM layer rule for {arch}")


def save_checkpoint(path: str | Path, model: nn.Module, arch: str, classes: list[str],
                    binary_map: dict[str, str], img_size: int, temperature: float = 1.0, **extra):
    torch.save({"arch": arch, "classes": classes, "binary_map": binary_map, "img_size": img_size,
                "temperature": temperature, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = create_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def unsafe_mask(classes: list[str], binary_map: dict[str, str]) -> torch.Tensor:
    return torch.tensor([binary_map[c] == "unsafe" for c in classes])
