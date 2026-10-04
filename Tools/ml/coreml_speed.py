"""Measures Core ML prediction latency for one package and compute unit (run as a subprocess so a
crash in one configuration is recorded instead of killing the whole evaluation).
Usage: coreml_speed.py <package> <CPU_ONLY|CPU_AND_GPU|CPU_AND_NE|ALL> <WxH> [<WxH> ...]
Prints one JSON object per input size."""
import json
import sys
import time

import coremltools as ct
import numpy as np

package, unit_name, sizes = sys.argv[1], sys.argv[2], sys.argv[3:]
start = time.time()
model = ct.models.MLModel(package, compute_units=getattr(ct.ComputeUnit, unit_name))
load_s = time.time() - start
for size in sizes:
    w, h = map(int, size.split("x"))
    x = np.random.rand(1, 3, h, w).astype(np.float16)
    t0 = time.perf_counter()
    model.predict({"input": x})  # first call per shape includes on-device compilation for it
    first_ms = (time.perf_counter() - t0) * 1000
    times, deadline = [], time.time() + 8
    while len(times) < 15 and (len(times) < 3 or time.time() < deadline):
        t0 = time.perf_counter()
        model.predict({"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    print(json.dumps({"input": size, "runs": len(times), "median_ms": float(np.median(times)),
                      "p90_ms": float(np.percentile(times, 90)), "first_ms": first_ms, "load_s": load_s}), flush=True)
