"""Stage 15 experiment: W8A8 (int8 weights + int8 activations) for the M4 Neural Engine.

The M4 Neural Engine has a faster int8×int8 path; whether it helps these models, and what it costs in
quality, is measured here rather than assumed. Activation calibration runs the model on the CPU, and
Core ML's CPU backend aborts on enumerated input shapes (Docs/02), so this experiment uses one
fixed-shape model per input size. It also answers a second question: is a fixed shape faster on the
Neural Engine than the enumerated-shape model the app ships?

For each model and input size: FP16 fixed shape vs W8A8 fixed shape —
  * speed on CPU+ANE (subprocess, coreml_speed.py), next to the shipped enumerated-shape model
  * quality: PSNR of W8A8 output vs FP16 output, and both vs ground truth (720p only)
Writes Docs/data/int8_eval.json. Temporary packages go to a temp dir.

Usage: .venv/bin/python Tools/ml/quantize_int8.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import coremltools.optimize as cto
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import convert_to_coreml as C  # noqa: E402
import evaluate_models as E  # noqa: E402
import sr_models  # noqa: E402

MODELS = [("span-nomosuni", 2), ("realesr-animevideov3", 2)]
SIZES = [(360, 640), (540, 960), (720, 1280)]


def calibration_samples(h: int, w: int, n: int = 12) -> list[dict]:
    rng = np.random.default_rng(1)
    samples = []
    while len(samples) < n:
        for path in E.IMAGES:
            img = E.resize(E.load_rgb(path), (2560, 1440), Image.LANCZOS)
            x0, y0 = rng.integers(0, 2560 - w), rng.integers(0, 1440 - h)
            crop = E.jpeg(img[y0:y0 + h, x0:x0 + w], 60)
            samples.append({"input": crop.transpose(2, 0, 1)[None].astype(np.float16)})
    return samples[:n]


def fixed_model(model_id: str, scale: int, h: int, w: int):
    spec = sr_models.SPECS[model_id]
    wrapper = C.CoreMLWrapper(sr_models.load(model_id), spec.native_scale, scale).eval()
    traced = torch.jit.trace(wrapper, torch.rand(1, 3, h, w))
    return ct.convert(traced, inputs=[ct.TensorType(name="input", shape=(1, 3, h, w), dtype=np.float16)],
                      outputs=[ct.TensorType(name="output", dtype=np.float16)], convert_to="mlprogram",
                      compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)


def speed(package: Path, size: str) -> float | None:
    proc = subprocess.run([sys.executable, str(Path(__file__).parent / "coreml_speed.py"), str(package), "CPU_AND_NE", size],
                          capture_output=True, text=True)
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    return rows[0]["median_ms"] if rows else None


def main() -> None:
    results = []
    tmp = Path(tempfile.mkdtemp(prefix="ve-int8-"))
    for model_id, scale in MODELS:
        shipped = E.ROOT / "Models" / f"{model_id}-x{scale}.mlpackage"
        for h, w in SIZES:
            size = f"{w}x{h}"
            fp16 = fixed_model(model_id, scale, h, w)
            act = cto.coreml.OptimizationConfig(
                global_config=cto.coreml.experimental.OpActivationLinearQuantizerConfig(mode="linear_symmetric"))
            a8 = cto.coreml.experimental.linear_quantize_activations(fp16, act, calibration_samples(h, w))
            w8a8 = cto.coreml.linear_quantize_weights(a8, cto.coreml.OptimizationConfig(
                global_config=cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")))
            for m, tag in [(fp16, "fp16"), (w8a8, "w8a8")]:
                m.user_defined_metadata.update({"ve.id": f"{model_id}-{size}-{tag}", "ve.family": model_id,
                                                "ve.scale": str(scale), "ve.displayName": f"{model_id} {tag}"})
            p16, p8 = tmp / f"{model_id}-{size}-fp16.mlpackage", tmp / f"{model_id}-{size}-w8a8.mlpackage"
            fp16.save(str(p16))
            w8a8.save(str(p8))

            row = {"model": model_id, "input": size,
                   "enumerated_fp16_ms": speed(shipped, size), "fixed_fp16_ms": speed(p16, size), "fixed_w8a8_ms": speed(p8, size)}

            # Quality: W8A8 vs FP16 on natural crops at this size, and vs ground truth at 720p.
            r16, r8 = E.CoreMLRunner(p16, ct.ComputeUnit.CPU_AND_NE), E.CoreMLRunner(p8, ct.ComputeUnit.CPU_AND_NE)
            agree, gt16, gt8 = [], [], []
            for path in E.IMAGES:
                gt = E.resize(E.load_rgb(path), (w * scale, h * scale), Image.LANCZOS)
                lr = E.jpeg(E.resize(gt, (w, h), Image.BICUBIC), 60)
                a, b = r8(lr), r16(lr)
                agree.append(E.psnr(E.to_luma(a), E.to_luma(b)))
                gt8.append(E.psnr(E.to_luma(a), E.to_luma(gt)))
                gt16.append(E.psnr(E.to_luma(b), E.to_luma(gt)))
            row |= {"w8a8_vs_fp16_psnr": float(np.mean(agree)), "gt_psnr_fp16": float(np.mean(gt16)), "gt_psnr_w8a8": float(np.mean(gt8))}
            results.append(row)
            print(f"[int8] {model_id} {size}: enumerated FP16 {row['enumerated_fp16_ms']} ms | fixed FP16 {row['fixed_fp16_ms']} ms | "
                  f"fixed W8A8 {row['fixed_w8a8_ms']} ms | W8A8 vs FP16 {row['w8a8_vs_fp16_psnr']:.1f} dB | "
                  f"GT {row['gt_psnr_fp16']:.2f} → {row['gt_psnr_w8a8']:.2f} dB", flush=True)

    (E.DOCS / "data" / "int8_eval.json").write_text(json.dumps(results, indent=2))
    print("saved Docs/data/int8_eval.json")


if __name__ == "__main__":
    main()
