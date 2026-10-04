"""PyTorch definitions/loaders for the Super Resolution models used by VideoEnhancer.

SRVGGNetCompact follows the architecture published with Real-ESRGAN (BSD-3-Clause, Xintao Wang et al.):
conv3x3 + PReLU blocks at input resolution, PixelShuffle, plus a nearest-upsampled copy of the input.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Models" / "source"


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_feat: int = 64, num_conv: int = 16, upscale: int = 4):
        super().__init__()
        self.upscale = upscale
        layers: list[nn.Module] = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            layers += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        layers.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(layers)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    architecture: str
    native_scale: int
    license: str
    source_repo: str
    params: int = 0


SPECS = {
    "realesr-animevideov3": ModelSpec(
        "realesr-animevideov3", "Real-ESRGAN AnimeVideo v3", "SRVGGNetCompact (64 feat, 16 conv)", 4,
        "BSD-3-Clause", "mlx-community/Real-ESRGAN-animevideov3"),
    "realesr-general-x4v3": ModelSpec(
        "realesr-general-x4v3", "Real-ESRGAN General v3", "SRVGGNetCompact (64 feat, 32 conv)", 4,
        "BSD-3-Clause", "clavidk/realesr-general-x4v3"),
    "span-nomosuni": ModelSpec(
        "span-nomosuni", "SPAN 2x NomosUni (multi-JPEG)", "SPAN", 2,
        "CC-BY-4.0", "Phips/2xNomosUni_span_multijpg"),
}


def load(model_id: str) -> nn.Module:
    """Returns the model in eval mode with fp32 weights."""
    if model_id == "realesr-animevideov3":
        from safetensors.numpy import load_file
        weights = load_file(SOURCE / "realesr-animevideov3.mlx.safetensors")
        model = SRVGGNetCompact(num_feat=64, num_conv=16, upscale=4)
        state = {}
        for key, value in weights.items():
            value = value.astype(np.float32)
            if value.ndim == 4:  # MLX conv layout (O, kH, kW, I) → PyTorch (O, I, kH, kW)
                value = np.transpose(value, (0, 3, 1, 2))
            state[key] = torch.from_numpy(np.ascontiguousarray(value))
        model.load_state_dict(state, strict=True)
    elif model_id == "realesr-general-x4v3":
        checkpoint = torch.load(SOURCE / "realesr-general-x4v3.pth", map_location="cpu", weights_only=True)
        model = SRVGGNetCompact(num_feat=64, num_conv=32, upscale=4)
        model.load_state_dict(checkpoint["params"], strict=True)
    elif model_id == "span-nomosuni":
        import spandrel
        descriptor = spandrel.ModelLoader().load_from_file(SOURCE / "2xNomosUni_span_multijpg.safetensors")
        assert descriptor.scale == 2, descriptor.scale
        model = descriptor.model
    else:
        raise KeyError(model_id)
    model.eval()
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
