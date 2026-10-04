"""Stage 6: download pretrained Super Resolution weights from Hugging Face.

Files land in Models/source/; Models/source/manifest.json records the exact repo revision (commit sha),
license and size of every file, so a converted Core ML model can always be traced back to its source.

Usage: .venv/bin/python Tools/ml/download_models.py
"""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "Models" / "source"

# (repo, file in repo, local name, license, role)
FILES = [
    ("mlx-community/Real-ESRGAN-animevideov3", "model.safetensors", "realesr-animevideov3.mlx.safetensors",
     "BSD-3-Clause", "weights (fp16, MLX layout) converted from the official xinntao release"),
    ("skillsafe-ai/realesr-animevideov3", "reference/input-64x64.npy", "realesr-animevideov3.ref-input.npy",
     "BSD-3-Clause", "reference input for validating the reconstruction"),
    ("skillsafe-ai/realesr-animevideov3", "reference/output-64x64.npy", "realesr-animevideov3.ref-output.npy",
     "BSD-3-Clause", "reference output for validating the reconstruction"),
    ("clavidk/realesr-general-x4v3", "realesr-general-x4v3.pth", "realesr-general-x4v3.pth",
     "BSD-3-Clause", "original PyTorch checkpoint from the xinntao release"),
    ("Phips/2xNomosUni_span_multijpg", "2xNomosUni_span_multijpg.safetensors", "2xNomosUni_span_multijpg.safetensors",
     "CC-BY-4.0", "SPAN x2 trained with JPEG + multiscale degradations (by Philip Hofmann)"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest = []
    for repo, filename, local_name, license_name, role in FILES:
        revision = api.model_info(repo).sha
        cached = Path(hf_hub_download(repo, filename, revision=revision))
        target = DEST / local_name
        target.write_bytes(cached.read_bytes())
        manifest.append({
            "repo": repo, "file": filename, "revision": revision, "local": local_name,
            "bytes": target.stat().st_size, "sha256": sha256(target), "license": license_name, "role": role,
            "url": f"https://huggingface.co/{repo}/blob/{revision}/{filename}",
        })
        print(f"{local_name:45s} {target.stat().st_size:>10,d} B  {repo}@{revision[:10]}")
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
