"""Stage 8: convert the PyTorch SR models to Core ML (.mlpackage, ML Program, FP16).

Model I/O contract (what Sources/VideoEnhancer/ML/CoreMLSuperResolution.swift expects):
  input  "input"  : Float16 [1, 3, H, W], RGB in [0, 1] (BT.709 gamma-encoded, as decoded video)
  output "output" : Float16 [1, 3, H*s, W*s], RGB clamped to [0, 1]
  H×W is one of the enumerated shapes (ANE-friendly; flexible ranges tend to fall back to GPU/CPU).
Variants with output scale 2 from ×4 networks average-pool the ×4 result 2×2 inside the model
(supersampled ×2, half the output bandwidth of ×4).

Usage: .venv/bin/python Tools/ml/convert_to_coreml.py [model-id ...]
"""
import json
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import sr_models  # noqa: E402

OUT = sr_models.ROOT / "Models"
SHAPES = [  # (H, W)
    (270, 480), (360, 640), (540, 960), (720, 1280), (1080, 1920),  # 16:9
    (268, 640), (400, 960), (536, 1280), (800, 1920),               # ~2.39:1 scope films
    (480, 640), (720, 960),                                          # 4:3
]
DEFAULT_SHAPE = (360, 640)
MODEL_VERSION = "1.0"

# model id → output scales to export
EXPORTS = {
    "realesr-animevideov3": [2, 4],
    "realesr-general-x4v3": [2, 4],
    "span-nomosuni": [2],
}
# "-flex" variants with RangeDim (flexible) input shapes, for the CPU-only compute unit: Core ML's CPU
# backend aborts (SIGTRAP) on ML Programs with enumerated input shapes on this system, while fixed and
# range shapes work (Docs/02). The Neural Engine prefers enumerated shapes, so those stay the default.
FLEX_EXPORTS = {"span-nomosuni": [2], "realesr-animevideov3": [2], "realesr-general-x4v3": [2]}


class CoreMLWrapper(nn.Module):
    def __init__(self, model: nn.Module, native_scale: int, out_scale: int):
        super().__init__()
        self.model = model
        self.pool = native_scale // out_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if self.pool > 1:
            y = F.avg_pool2d(y, self.pool)
        return torch.clamp(y, 0.0, 1.0)


def convert(model_id: str, out_scale: int, flex: bool = False) -> Path:
    spec = sr_models.SPECS[model_id]
    model = sr_models.load(model_id)
    wrapper = CoreMLWrapper(model, spec.native_scale, out_scale).eval()
    example = torch.rand(1, 3, *DEFAULT_SHAPE)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    start = time.time()
    if flex:
        shape = ct.Shape(shape=(1, 3, ct.RangeDim(lower_bound=64, upper_bound=1080, default=DEFAULT_SHAPE[0]),
                                ct.RangeDim(lower_bound=64, upper_bound=1920, default=DEFAULT_SHAPE[1])))
    else:
        shape = ct.EnumeratedShapes(shapes=[(1, 3, h, w) for h, w in SHAPES], default=(1, 3, *DEFAULT_SHAPE))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    convert_seconds = time.time() - start

    manifest = {m["local"]: m for m in json.loads((sr_models.SOURCE / "manifest.json").read_text())}
    source = next(m for m in manifest.values() if m["repo"] == spec.source_repo and m["local"].endswith((".pth", ".safetensors")))
    used_params = sum(p.numel() for name, p in model.named_parameters()
                      if not any(t in name for t in (".sk.", ".conv.0", ".conv.1", ".conv.2")) or "eval_conv" in name)

    mlmodel.author = f"{spec.display_name} — weights from huggingface.co/{spec.source_repo}"
    mlmodel.license = spec.license
    mlmodel.short_description = (f"{spec.display_name}: {spec.architecture}, ×{out_scale} output "
                                 f"(network ×{spec.native_scale}). Input/output Float16 RGB [0,1], NCHW"
                                 + (", flexible input size (CPU-only variant)." if flex else "."))
    mlmodel.version = MODEL_VERSION
    mlmodel.user_defined_metadata.update({
        "ve.id": f"{model_id}-x{out_scale}" + ("-flex" if flex else ""),
        "ve.shapeMode": "range" if flex else "enumerated",
        "ve.family": model_id,
        "ve.displayName": spec.display_name,
        "ve.architecture": spec.architecture,
        "ve.scale": str(out_scale),
        "ve.nativeScale": str(spec.native_scale),
        "ve.inputShapes": ",".join(f"{w}x{h}" for h, w in SHAPES),
        "ve.temporalFrames": "1",
        "ve.sourceRepo": spec.source_repo,
        "ve.sourceRevision": source["revision"],
        "ve.sourceSHA256": source["sha256"],
        "ve.parameters": str(used_params),
    })

    path = OUT / (f"{model_id}-x{out_scale}" + ("-flex" if flex else "") + ".mlpackage")
    mlmodel.save(str(path))
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"{path.name:40s} {size / 1e6:6.2f} MB  converted in {convert_seconds:5.1f} s")
    return path


def main() -> None:
    ids = sys.argv[1:] or list(EXPORTS)
    for model_id in ids:
        for scale in EXPORTS[model_id]:
            convert(model_id, scale)
        for scale in FLEX_EXPORTS.get(model_id, []):
            convert(model_id, scale, flex=True)


if __name__ == "__main__":
    main()
