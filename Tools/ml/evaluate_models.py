"""Stage 7: test the converted Core ML models on their own, outside the player.

Measures, for every model in Models/*.mlpackage with output scale 2:
  1. Correctness  — Core ML (FP16) output vs the PyTorch FP32 reference (PSNR between them).
  2. Quality      — against ground truth: 4K photos → 2560×1440 GT → 1280×720 LR (bicubic),
                    clean and with JPEG q45 (stand-in for codec artifacts). PSNR/SSIM on luma and a
                    detail ratio (gradient energy SR / GT: <1 blurry, >1 over-sharpened/hallucinated).
                    Classical baselines: bilinear, bicubic, Lanczos.
  3. Temporal stability — 24 frames of sub-pixel panning; flicker = mean |ΔSR − ΔGT| between
                    consecutive frames (lower is better). Image SR runs frame by frame, so this shows
                    how much it "boils".
  4. Speed        — Core ML prediction latency per compute unit and input size (median / p90).
                    Includes Python ↔ Core ML copies, so the in-app benchmark (zero-copy) is faster.

Results: Docs/data/model_eval.json, Docs/02_MODEL_EVALUATION.md, Docs/images/*.png
Usage: .venv/bin/python Tools/ml/evaluate_models.py [--quick]
"""
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import sr_models  # noqa: E402

ROOT = sr_models.ROOT
IMAGES = sorted((ROOT / "TestMedia" / "images").glob("*.png"))
DOCS = ROOT / "Docs"
QUICK = "--quick" in sys.argv

COMPUTE_UNITS = {
    "CPU": ct.ComputeUnit.CPU_ONLY,
    "CPU+GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU+ANE": ct.ComputeUnit.CPU_AND_NE,
    "ALL": ct.ComputeUnit.ALL,
}
SPEED_SHAPES = [(360, 640), (540, 960), (720, 1280)]


# MARK: - Image helpers (float32 HWC RGB in [0, 1])

def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def resize(img: np.ndarray, size: tuple[int, int], method) -> np.ndarray:
    pil = Image.fromarray(np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return np.asarray(pil.resize(size, method), dtype=np.float32) / 255.0


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=quality)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0


def to_luma(img: np.ndarray) -> np.ndarray:
    return img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    x = torch.from_numpy(a)[None, None]
    y = torch.from_numpy(b)[None, None]
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :])[None, None]
    mu_x, mu_y = F.conv2d(x, w), F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mu_x ** 2
    syy = F.conv2d(y * y, w) - mu_y ** 2
    sxy = F.conv2d(x * y, w) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return float(s.mean())


def gradient_energy(y: np.ndarray) -> float:
    gx = np.diff(y, axis=1)[:-1, :]
    gy = np.diff(y, axis=0)[:, :-1]
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


# MARK: - Model runners

class CoreMLRunner:
    def __init__(self, path: Path, units=ct.ComputeUnit.ALL):
        start = time.time()
        self.model = ct.models.MLModel(str(path), compute_units=units)
        self.load_seconds = time.time() - start
        meta = self.model.user_defined_metadata
        self.id = meta["ve.id"]
        self.family = meta["ve.family"]
        self.scale = int(meta["ve.scale"])
        self.name = meta["ve.displayName"]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None]).astype(np.float16)
        y = self.model.predict({"input": x})["output"]
        return np.asarray(y[0], dtype=np.float32).transpose(1, 2, 0)


def torch_reference(family: str, scale: int):
    model = sr_models.load(family)
    native = sr_models.SPECS[family].native_scale

    def run(img: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            y = model(torch.from_numpy(img.transpose(2, 0, 1)[None].copy()))
            if native // scale > 1:
                y = F.avg_pool2d(y, native // scale)
            return y.clamp(0, 1)[0].numpy().transpose(1, 2, 0)
    return run


# MARK: - Test data

def make_pairs() -> list[dict]:
    """GT 2560×1440 from each 4K photo, LR 1280×720 clean and JPEG-degraded."""
    pairs = []
    for path in IMAGES:
        gt = resize(load_rgb(path), (2560, 1440), Image.LANCZOS)
        lr = resize(gt, (1280, 720), Image.BICUBIC)
        pairs.append({"name": path.stem, "degradation": "bicubic", "gt": gt, "lr": lr})
        pairs.append({"name": path.stem, "degradation": "bicubic+jpeg45", "gt": gt, "lr": jpeg(lr, 45)})
    return pairs


def make_pan_sequence(frames: int = 24) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Sub-pixel pan (0.37 GT px/frame) over a textured region; LR 640×360, GT 1280×720."""
    src = load_rgb(IMAGES[0])
    region = resize(src[600:1800, 200:2333], (1600, 900), Image.LANCZOS)  # fixed crop, then fit
    gts, lrs = [], []
    pil = Image.fromarray(np.clip(region * 255 + 0.5, 0, 255).astype(np.uint8))
    for t in range(frames):
        dx = 0.37 * t
        crop = pil.transform((1280, 720), Image.AFFINE, (1, 0, 100 + dx, 0, 1, 80), resample=Image.BICUBIC)
        gt = np.asarray(crop, dtype=np.float32) / 255.0
        gts.append(gt)
        lrs.append(jpeg(resize(gt, (640, 360), Image.BICUBIC), 60))
    return gts, lrs


CLASSICAL = {
    "Bilinear": lambda lr: resize(lr, (lr.shape[1] * 2, lr.shape[0] * 2), Image.BILINEAR),
    "Bicubic": lambda lr: resize(lr, (lr.shape[1] * 2, lr.shape[0] * 2), Image.BICUBIC),
    "Lanczos": lambda lr: resize(lr, (lr.shape[1] * 2, lr.shape[0] * 2), Image.LANCZOS),
}


def quality_row(sr: np.ndarray, gt: np.ndarray) -> dict:
    ys, yg = to_luma(sr), to_luma(gt)
    return {"psnr": psnr(ys, yg), "ssim": ssim(ys, yg), "detail": gradient_energy(ys) / gradient_energy(yg)}


def flicker(srs: list[np.ndarray], gts: list[np.ndarray]) -> float:
    errs = [np.mean(np.abs((to_luma(srs[t]) - to_luma(srs[t - 1])) - (to_luma(gts[t]) - to_luma(gts[t - 1]))))
            for t in range(1, len(srs))]
    return float(np.mean(errs)) * 255.0  # in 8-bit code values


def save_crops(name: str, outputs: dict[str, np.ndarray], gt: np.ndarray, box: tuple[int, int, int, int]) -> str:
    x, y, w, h = box
    tiles = [("Ground truth", gt)] + list(outputs.items())
    zoom = 2
    canvas = Image.new("RGB", (w * zoom * len(tiles), h * zoom + 22), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for i, (label, img) in enumerate(tiles):
        crop = Image.fromarray(np.clip(img[y:y + h, x:x + w] * 255 + 0.5, 0, 255).astype(np.uint8))
        canvas.paste(crop.resize((w * zoom, h * zoom), Image.NEAREST), (i * w * zoom, 22))
        draw.text((i * w * zoom + 6, 4), label, fill="black")
    out = DOCS / "images" / f"sr_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return str(out.relative_to(DOCS))


# MARK: - Main

def main() -> None:
    packages = sorted(p for p in (ROOT / "Models").glob("*.mlpackage"))
    runners = [CoreMLRunner(p) for p in packages]
    x2 = [r for r in runners if r.scale == 2]
    results: dict = {"machine": machine_info(), "models": {}, "quality": [], "flicker": {}, "speed": []}

    # 1. Core ML vs PyTorch reference.
    probe = resize(load_rgb(IMAGES[0]), (640, 360), Image.LANCZOS)
    for r in x2:
        ref = torch_reference(r.family, r.scale)(probe)
        results["models"][r.id] = {
            "name": r.name, "family": r.family, "scale": r.scale,
            "coreml_vs_pytorch_psnr": psnr(r(probe), ref),
            "package_mb": sum(f.stat().st_size for f in (ROOT / "Models" / f"{r.id}.mlpackage").rglob("*") if f.is_file()) / 1e6,
        }
        print(f"[correctness] {r.id}: Core ML FP16 vs PyTorch FP32 PSNR {results['models'][r.id]['coreml_vs_pytorch_psnr']:.1f} dB")

    # 2. Quality against ground truth.
    pairs = make_pairs()
    for pair in pairs:
        outputs = {name: fn(pair["lr"]) for name, fn in CLASSICAL.items()}
        for r in x2:
            outputs[r.name] = r(pair["lr"])
        for method, sr in outputs.items():
            row = quality_row(sr, pair["gt"]) | {"image": pair["name"], "degradation": pair["degradation"], "method": method}
            results["quality"].append(row)
        if pair["degradation"] == "bicubic+jpeg45":
            box = (1100, 560, 240, 160) if "aerial" in pair["name"] else (1200, 700, 240, 160)
            results.setdefault("images", []).append(save_crops(pair["name"], {k: outputs[k] for k in ["Bicubic"] + [r.name for r in x2]}, pair["gt"], box))
        print(f"[quality] {pair['name']} / {pair['degradation']} done")

    # 3. Temporal stability.
    gts, lrs = make_pan_sequence(12 if QUICK else 24)
    for name, fn in [("Bicubic", CLASSICAL["Bicubic"])] + [(r.name, r) for r in x2]:
        results["flicker"][name] = flicker([fn(lr) for lr in lrs], gts)
        print(f"[flicker] {name}: {results['flicker'][name]:.2f}")

    # 5. Does AI help if the input must be reduced for speed? 720p JPEG source → 1440p GT.
    by_id = {r.id: r for r in runners}
    experiments = {
        "Lanczos 720p→1440p (classical)": lambda s: resize(s, (2560, 1440), Image.LANCZOS),
        "AI full input: SPAN ×2": lambda s: by_id["span-nomosuni-x2"](s),
        "AI full input: General v3 ×2": lambda s: by_id["realesr-general-x4v3-x2"](s),
        "AI full input: AnimeVideo v3 ×2": lambda s: by_id["realesr-animevideov3-x2"](s),
        "360p input: AnimeVideo v3 ×4": lambda s: by_id["realesr-animevideov3-x4"](resize(s, (640, 360), Image.BOX)),
        "360p input: General v3 ×4": lambda s: by_id["realesr-general-x4v3-x4"](resize(s, (640, 360), Image.BOX)),
        "540p input: General v3 ×2 + Lanczos": lambda s: resize(by_id["realesr-general-x4v3-x2"](resize(s, (960, 540), Image.BOX)), (2560, 1440), Image.LANCZOS),
    }
    for path in IMAGES:
        gt = resize(load_rgb(path), (2560, 1440), Image.LANCZOS)
        source = jpeg(resize(gt, (1280, 720), Image.BICUBIC), 60)
        for name, fn in experiments.items():
            results.setdefault("reduced_input", []).append(quality_row(fn(source), gt) | {"image": path.stem, "method": name})
    print("[reduced input] done")

    # 4. Speed per compute unit (subprocess per configuration, see coreml_speed.py).
    for package in packages:
        meta = ct.models.MLModel(str(package), skip_model_load=True).user_defined_metadata
        if int(meta["ve.scale"]) != 2:
            continue
        for unit_name, unit in COMPUTE_UNITS.items():
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).parent / "coreml_speed.py"), str(package), unit.name,
                 *[f"{w}x{h}" for h, w in SPEED_SHAPES]],
                capture_output=True, text=True)
            rows = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
            for row in rows:
                row |= {"model": meta["ve.id"], "units": unit_name}
                results["speed"].append(row)
                print(f"[speed] {row['model']:28s} {unit_name:8s} {row['input']}: median {row['median_ms']:8.1f} ms  p90 {row['p90_ms']:8.1f} ms")
            if proc.returncode != 0:
                failure = {"model": meta["ve.id"], "units": unit_name, "error": f"process exited with {proc.returncode}"
                           + (" (SIGTRAP crash inside Core ML)" if proc.returncode in (-5, 133) else "")}
                results["speed"].append(failure)
                print(f"[speed] {meta['ve.id']:28s} {unit_name:8s} FAILED: {failure['error']}")

    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "data" / "model_eval.json").write_text(json.dumps(results, indent=2))
    print("saved Docs/data/model_eval.json")


def machine_info() -> dict:
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return {"chip": chip, "macos": platform.mac_ver()[0], "coremltools": ct.__version__, "torch": torch.__version__}


if __name__ == "__main__":
    main()
