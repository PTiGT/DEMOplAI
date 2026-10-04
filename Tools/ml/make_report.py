"""Builds the measurement tables in Docs/02_MODEL_EVALUATION.md and Docs/03_PERFORMANCE.md from the
JSON files in Docs/data/ (so no number in the docs is typed by hand).

Usage: .venv/bin/python Tools/ml/make_report.py
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Docs" / "data"


def load(name):
    path = DATA / name
    return json.loads(path.read_text()) if path.exists() else None


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def f(v, digits=1):
    return "—" if v is None else f"{v:.{digits}f}"


def model_eval_section() -> str:
    d = load("model_eval.json")
    if not d:
        return "_model_eval.json missing — run Tools/ml/evaluate_models.py_"
    parts = [f"Machine: {d['machine']['chip']}, macOS {d['machine']['macos']}, coremltools {d['machine']['coremltools']}, torch {d['machine']['torch']}.\n"]

    parts.append("### Core ML conversion correctness (FP16 Core ML vs FP32 PyTorch, same input)\n")
    parts.append(table(["Model", "PSNR Core ML vs PyTorch", "Package"],
                       [[m["name"] + f" ×{m['scale']}", f"{m['coreml_vs_pytorch_psnr']:.1f} dB", f"{m['package_mb']:.2f} MB"]
                        for m in d["models"].values()]))

    parts.append("\n### Quality vs ground truth (2560×1440 GT, 1280×720 input, ×2; mean of 3 photos)\n")
    parts.append("PSNR/SSIM on luma: fidelity to the original. Detail = gradient energy of the result ÷ ground truth "
                 "(< 1 softer than the original, > 1 sharper/hallucinated).\n")
    agg = collections.defaultdict(list)
    for r in d["quality"]:
        agg[(r["degradation"], r["method"])].append(r)
    for deg in ["bicubic", "bicubic+jpeg45"]:
        rows = []
        for (dg, method), rs in agg.items():
            if dg == deg:
                n = len(rs)
                rows.append([method, f(sum(x["psnr"] for x in rs) / n, 2), f(sum(x["ssim"] for x in rs) / n, 4),
                             f(sum(x["detail"] for x in rs) / n, 2)])
        parts.append(f"\n**Input degradation: {deg}**\n\n" + table(["Method", "PSNR dB", "SSIM", "Detail"], rows))

    if d.get("reduced_input"):
        parts.append("\n### Does AI help when the input must be reduced for speed? (720p JPEG source → 1440p GT)\n")
        agg2 = collections.defaultdict(list)
        for r in d["reduced_input"]:
            agg2[r["method"]].append(r)
        parts.append(table(["Method", "PSNR dB", "SSIM", "Detail"],
                           [[m, f(sum(x["psnr"] for x in rs) / len(rs), 2), f(sum(x["ssim"] for x in rs) / len(rs), 4),
                             f(sum(x["detail"] for x in rs) / len(rs), 2)] for m, rs in agg2.items()]))

    parts.append("\n### Temporal flicker on sub-pixel pan (mean |ΔSR − ΔGT| between frames, 8-bit units, lower is better)\n")
    parts.append(table(["Method", "Flicker"], [[k, f(v, 2)] for k, v in d["flicker"].items()]))

    parts.append("\n### Core ML latency, standalone (Python coremltools, includes array copies; median ms)\n")
    speed = collections.defaultdict(dict)
    failures = []
    for r in d["speed"]:
        if "error" in r:
            failures.append(f"{r['model']} / {r['units']}: {r['error']}")
            continue
        speed[(r["model"], r["units"])][r["input"]] = r["median_ms"]
    inputs = ["640x360", "960x540", "1280x720"]
    parts.append(table(["Model", "Compute"] + inputs,
                       [[m, u] + [f(v.get(i)) for i in inputs] for (m, u), v in speed.items()]))
    if failures:
        parts.append("\nFailed configurations: " + "; ".join(failures))
    return "\n".join(parts)


def postprocessing_section() -> str:
    d = load("postprocessing_eval.json")
    if not d:
        return "_postprocessing_eval.json missing — run Tools/ml/test_postprocessing.py_"
    agg = collections.defaultdict(list)
    for r in d["consistency"]:
        agg[(r["model"], r["variant"])].append(r)
    rows = [[m, v, f(sum(x["psnr"] for x in rs) / len(rs), 2), f(sum(x["ssim"] for x in rs) / len(rs), 4),
             f(sum(x["detail"] for x in rs) / len(rs), 2), rs[0]["mean_rgb_bias"]] for (m, v), rs in agg.items()]
    out = ["### Low-frequency consistency correction (720p JPEG source → 1440p GT)\n",
           table(["Model", "Variant", "PSNR dB", "SSIM", "Detail", "Mean RGB bias (8-bit, first photo)"], rows),
           "\n### Temporal stabilization: static scene, codec noise differs every frame (lower flicker is better)\n",
           table(["Model", "Variant", "Static flicker", "PSNR dB"],
                 [[r["model"], r["variant"], f(r["static_flicker"], 2), f(r.get("psnr"), 2)] for r in d["stabilization"]])]
    return "\n".join(out)


def int8_section() -> str:
    d = load("int8_eval.json")
    if not d:
        return "_int8_eval.json missing — run Tools/ml/quantize_int8.py_"
    rows = [[r["model"], r["input"], f(r["enumerated_fp16_ms"]), f(r["fixed_fp16_ms"]), f(r["fixed_w8a8_ms"]),
             f"{(1 - r['fixed_w8a8_ms'] / r['fixed_fp16_ms']) * 100:+.0f}%".replace("+-", "−").replace("+", "−") if r["fixed_w8a8_ms"] <= r["fixed_fp16_ms"]
             else f"+{(r['fixed_w8a8_ms'] / r['fixed_fp16_ms'] - 1) * 100:.0f}%",
             f(r["w8a8_vs_fp16_psnr"]), f(r["gt_psnr_fp16"], 2), f(r["gt_psnr_w8a8"], 2)] for r in d]
    return ("CPU+ANE, median ms (Python coremltools, subprocess per model). W8A8 = int8 weights + int8 activations, "
            "calibrated on natural photo crops.\n\n"
            + table(["Model", "Input", "Enumerated FP16 (shipped)", "Fixed FP16", "Fixed W8A8", "W8A8 time vs FP16",
                     "W8A8 vs FP16 PSNR", "GT PSNR FP16", "GT PSNR W8A8"], rows))


def benchmark_section() -> str:
    parts = []
    for name, title in [("benchmark_units.json", "Compute units (2 frames in flight, frames arriving at 60 fps)"),
                        ("benchmark_inflight1.json", "1 frame in flight (CPU+ANE)"),
                        ("benchmark_cpu.json", "CPU only")]:
        d = load(name)
        if d is None:
            parts.append(f"\n**{title}**: no data ({name} missing — the run may have crashed).")
            continue
        rows = [[r["configuration"].get("modelID") or "metal (Lanczos+CAS)",
                 "GPU" if not r["configuration"].get("modelID") else r["configuration"]["computeUnits"],
                 f"{r['configuration']['inputSize']['width']}×{r['configuration']['inputSize']['height']} → {r['outputSize']['width']}×{r['outputSize']['height']}",
                 f(r["averageFPS"]), f"{f(r.get('inferenceMedianMs'))} / {f(r.get('inferenceP95Ms'))}", f(r.get("totalLatencyMedianMs")),
                 f"{r['framesDropped']}/{r['framesOffered']}", r["peakMemoryBytes"] // 1048576,
                 f(r.get("cpuPercent"), 0), f(r.get("gpuBusyPercent"), 0), r.get("error") or ""]
                for r in d]
        note = ("\n\nWith 2 frames in flight the prediction wall time includes waiting for the other frame on the "
                "Neural Engine; pure inference time is in the 1-frame table. Peak MB accumulates across runs in one process."
                if name == "benchmark_units.json" else "")
        parts.append(f"\n**{title}**{note}\n\n" + table(
            ["Model", "Compute", "Input → Output", "Avg FPS", "Inference med / p95 ms", "Total ms", "Dropped", "Peak MB", "CPU %", "GPU %", "Error"], rows))
    return "\n".join(parts)


def replace_block(path: Path, marker: str, content: str) -> None:
    text = path.read_text()
    start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    if start not in text:
        raise SystemExit(f"{path.name}: marker {marker} missing")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{content}\n{end}{tail}")


if __name__ == "__main__":
    docs = ROOT / "Docs"
    replace_block(docs / "02_MODEL_EVALUATION.md", "model-eval", model_eval_section())
    replace_block(docs / "02_MODEL_EVALUATION.md", "postprocessing", postprocessing_section())
    replace_block(docs / "03_PERFORMANCE.md", "int8", int8_section())
    replace_block(docs / "03_PERFORMANCE.md", "benchmarks", benchmark_section())
    print("updated Docs/02_MODEL_EVALUATION.md and Docs/03_PERFORMANCE.md")
