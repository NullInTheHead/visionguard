"""Frozen pretrained backbones behind one interface.

Every backbone takes images in [0, 1] (as produced by ``data.Preprocess``) and returns
two intermediate feature maps. The first map sets the patch grid; input sizes are
chosen so every real backbone gives a 28x28 grid (protocol section 3).

Requires the ``models`` extra (torch, timm).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .data import Preprocess


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    timm_name: str
    kind: str  # "cnn", "vit" or "toy"
    layers: tuple[int, ...]  # timm feature indices (cnn) or 0-based block indices (vit)
    resize: int
    crop: int

    @property
    def preprocess(self) -> Preprocess:
        return Preprocess(resize=self.resize, crop=self.crop)


BACKBONES: dict[str, BackboneSpec] = {
    # timm "tv_in1k" = torchvision IMAGENET1K_V1 weights, the ones the PatchCore paper used.
    "wrn50": BackboneSpec("wrn50", "wide_resnet50_2.tv_in1k", "cnn", (2, 3), 256, 224),
    "vit_sup": BackboneSpec(
        "vit_sup", "vit_base_patch16_224.augreg2_in21k_ft_in1k", "vit", (3, 7), 512, 448
    ),
    "dinov2": BackboneSpec("dinov2", "vit_base_patch14_dinov2.lvd142m", "vit", (3, 7), 448, 392),
    "clip": BackboneSpec("clip", "vit_base_patch16_clip_224.openai", "vit", (3, 7), 512, 448),
    # Tiny random CNN for tests: no download, 7x7 grid at 56 px.
    "toy": BackboneSpec("toy", "", "toy", (0, 1), 64, 56),
}


class _ToyBackbone(nn.Module):
    """Fixed random conv net with outputs at stride 8 and 16 (tests only)."""

    def __init__(self) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(0)

        def conv(cin: int, cout: int) -> nn.Conv2d:
            layer = nn.Conv2d(cin, cout, 3, stride=2, padding=1)
            with torch.no_grad():
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=gen) / (3 * cin**0.5))
                layer.bias.zero_()
            return layer

        self.stem = nn.Sequential(conv(3, 16), nn.ReLU(), conv(16, 32), nn.ReLU())
        self.block1 = nn.Sequential(conv(32, 48), nn.ReLU())
        self.block2 = nn.Sequential(conv(48, 64), nn.ReLU())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.block1(self.stem(x))
        return [f1, self.block2(f1)]


class FeatureExtractor(nn.Module):
    """Frozen backbone: images in [0, 1], shape (B, 3, S, S) -> list of (B, C, h, w) maps."""

    def __init__(self, spec: BackboneSpec | str, pretrained: bool = True) -> None:
        super().__init__()
        self.spec = BACKBONES[spec] if isinstance(spec, str) else spec
        s = self.spec
        if s.kind == "toy":
            self.model = _ToyBackbone()
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        else:
            import timm

            if s.kind == "cnn":
                self.model = timm.create_model(
                    s.timm_name, pretrained=pretrained, features_only=True, out_indices=s.layers
                )
            elif s.kind == "vit":
                # img_size makes timm resample the position embeddings to the new grid.
                self.model = timm.create_model(s.timm_name, pretrained=pretrained, img_size=s.crop)
            else:
                raise ValueError(f"Unknown backbone kind: {s.kind}")
            cfg = self.model.pretrained_cfg
            mean, std = tuple(cfg["mean"]), tuple(cfg["std"])
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def preprocess(self) -> Preprocess:
        return self.spec.preprocess

    def train(self, mode: bool = True) -> FeatureExtractor:
        # Always frozen: BatchNorm and dropout stay in inference mode.
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        x = (images - self.mean) / self.std
        if self.spec.kind == "vit":
            return list(
                self.model.forward_intermediates(
                    x, indices=list(self.spec.layers), intermediates_only=True
                )
            )
        return list(self.model(x))


def pick_device(device: str = "auto") -> torch.device:
    """``auto`` picks CUDA, then Apple MPS, then CPU."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
