#!/bin/bash
# Runs the in-app benchmark matrix headlessly (same code path as playback) and stores JSON results
# in Docs/data/. Usage: scripts/run_benchmarks.sh [duration-seconds]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DURATION="${1:-5}"
swift build -c release --arch arm64 >/dev/null
BIN="$(swift build -c release --arch arm64 --show-bin-path)/VideoEnhancer"
OUT="$ROOT/Docs/data"
mkdir -p "$OUT"
MODELS="span-nomosuni-x2,realesr-animevideov3-x2,realesr-general-x4v3-x2"
INPUTS="640x360,960x540,1280x720"

echo "== Compute units (2 frames in flight, frames arriving at 60 fps)"
"$BIN" --benchmark --model "$MODELS,metal" --units cpuAndNeuralEngine,all,cpuAndGPU --input "$INPUTS" \
    --duration "$DURATION" --fps 60 --inflight 2 --json "$OUT/benchmark_units.json"

echo "== Frames in flight = 1 (CPU+ANE)"
"$BIN" --benchmark --model "$MODELS" --units cpuAndNeuralEngine --input "$INPUTS" \
    --duration "$DURATION" --fps 60 --inflight 1 --json "$OUT/benchmark_inflight1.json"

echo "== CPU only (separate process: Core ML may abort with Float16 I/O on CPU)"
"$BIN" --benchmark --model span-nomosuni-x2 --units cpuOnly --input 640x360 \
    --duration "$DURATION" --fps 60 --inflight 1 --json "$OUT/benchmark_cpu.json"
echo "cpuOnly exit code: $?"
