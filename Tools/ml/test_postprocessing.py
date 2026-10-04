"""Measures the two AI post-processing steps the app applies (numpy re-implementations of the
Metal kernels in ML.metal) before enabling them by default:
  * low-frequency consistency correction (consistencyResidual + consistencyApply)
  * temporal stabilization (temporalStabilize) on a static scene with frame-varying codec noise.
Writes Docs/data/postprocessing_eval.json.
Usage: .venv/bin/python Tools/ml/test_postprocessing.py"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import evaluate_models as E  # noqa: E402

rng = np.random.default_rng(0)


def box_down(img, s):
    h, w, c = img.shape
    return img.reshape(h // s, s, w // s, s, c).mean(axis=(1, 3))


def upsample_bicubic(img, size):
    return np.stack([np.asarray(Image.fromarray(img[..., k]).resize(size, Image.BICUBIC)) for k in range(3)], -1)


def tent_blur(img):
    k = np.array([0.25, 0.5, 0.25])
    out = img.copy()
    for axis in (0, 1):
        p = np.pad(out, [(1, 1) if a == axis else (0, 0) for a in range(3)], mode="edge")
        sl = [slice(None)] * 3
        acc = 0
        for i, kv in enumerate(k):
            sl[axis] = slice(i, i + out.shape[axis])
            acc = acc + kv * p[tuple(sl)]
        out = acc
    return out


def consistency(sr, lr, strength=1.0):
    s = sr.shape[0] // lr.shape[0]
    residual = tent_blur((lr - box_down(sr, s)).astype(np.float32))
    return np.clip(sr + strength * upsample_bicubic(residual, (sr.shape[1], sr.shape[0])), 0, 1)


def stabilize(srs, lrs, strength=0.5):
    out = [srs[0]]
    for t in range(1, len(srs)):
        change = np.abs(E.to_luma(lrs[t]) - E.to_luma(lrs[t - 1]))
        pad = np.pad(change, 1, mode="edge")
        local = np.max([pad[1 + dy:1 + dy + change.shape[0], 1 + dx:1 + dx + change.shape[1]] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
        x = np.clip((local - 1.5 / 255) / (4.5 / 255), 0, 1)
        still = 1 - x * x * (3 - 2 * x)
        still = np.asarray(Image.fromarray(still.astype(np.float32)).resize((srs[t].shape[1], srs[t].shape[0]), Image.BILINEAR))[..., None]
        out.append(srs[t] * (1 - strength * still) + out[-1] * strength * still)
    return out


def main():
    runners = {p: E.CoreMLRunner(E.ROOT / "Models" / f"{p}.mlpackage")
               for p in ["span-nomosuni-x2", "realesr-animevideov3-x2", "realesr-general-x4v3-x2"]}
    results = {"consistency": [], "stabilization": []}

    for path in E.IMAGES:
        gt = E.resize(E.load_rgb(path), (2560, 1440), Image.LANCZOS)
        lr = E.jpeg(E.resize(gt, (1280, 720), Image.BICUBIC), 60)
        for pid, run in runners.items():
            sr = run(lr)
            for label, img in [("raw", sr), ("consistency", consistency(sr, lr))]:
                row = E.quality_row(img, gt) | {"model": pid, "variant": label, "image": path.stem,
                                                "mean_rgb_bias": ((img - gt).reshape(-1, 3).mean(0) * 255).round(2).tolist()}
                results["consistency"].append(row)

    # Static scene, codec noise changes every frame (re-encoded with tiny noise): GT difference is 0,
    # so any frame-to-frame change of the output is flicker.
    gt = E.resize(E.load_rgb(E.IMAGES[1]), (1280, 720), Image.LANCZOS)
    base = E.resize(gt, (640, 360), Image.BICUBIC)
    lrs = [E.jpeg(np.clip(base + rng.normal(0, 1.0 / 255, base.shape).astype(np.float32), 0, 1), 60) for _ in range(16)]
    for pid, run in runners.items():
        srs = [run(lr) for lr in lrs]
        for label, seq in [("raw", srs), ("consistency", [consistency(s, l) for s, l in zip(srs, lrs)]),
                           ("stabilized", stabilize(srs, lrs)),
                           ("consistency+stabilized", stabilize([consistency(s, l) for s, l in zip(srs, lrs)], lrs))]:
            flick = float(np.mean([np.mean(np.abs(E.to_luma(seq[t]) - E.to_luma(seq[t - 1]))) for t in range(1, len(seq))]) * 255)
            q = E.quality_row(seq[-1], gt)
            results["stabilization"].append({"model": pid, "variant": label, "static_flicker": flick} | q)
    bic = [E.CLASSICAL["Bicubic"](lr) for lr in lrs]
    results["stabilization"].append({"model": "Bicubic", "variant": "raw",
        "static_flicker": float(np.mean([np.mean(np.abs(E.to_luma(bic[t]) - E.to_luma(bic[t - 1]))) for t in range(1, len(bic))]) * 255)})

    (E.DOCS / "data" / "postprocessing_eval.json").write_text(json.dumps(results, indent=2))
    import collections
    agg = collections.defaultdict(list)
    for r in results["consistency"]:
        agg[(r["model"], r["variant"])].append(r)
    for (m, v), rows in agg.items():
        n = len(rows)
        print(f"[consistency] {m:26s} {v:12s} PSNR {sum(r['psnr'] for r in rows)/n:6.2f} SSIM {sum(r['ssim'] for r in rows)/n:.4f} "
              f"detail {sum(r['detail'] for r in rows)/n:.2f} bias {rows[0]['mean_rgb_bias']}")
    for r in results["stabilization"]:
        print(f"[stabilization] {r['model']:26s} {r['variant']:24s} static flicker {r['static_flicker']:.2f}"
              + (f"  PSNR {r['psnr']:.2f} SSIM {r['ssim']:.4f}" if "psnr" in r else ""))


if __name__ == "__main__":
    main()
