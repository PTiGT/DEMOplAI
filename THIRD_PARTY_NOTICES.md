# Third-party notices

The Core ML models in `Models/` are converted from third-party weights (see `Models/source/manifest.json`
for exact repositories, revisions and SHA256 of the source files).

## Real-ESRGAN (realesr-animevideov3, realesr-general-x4v3)

Weights from https://github.com/xinntao/Real-ESRGAN, obtained via Hugging Face
(`mlx-community/Real-ESRGAN-animevideov3`, `clavidk/realesr-general-x4v3`).

```
BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## SPAN 2xNomosUni_span_multijpg

Model by Philip Hofmann (https://huggingface.co/Phips/2xNomosUni_span_multijpg), licensed under
Creative Commons Attribution 4.0 International (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).
Changes: converted to Core ML (FP16, enumerated and flexible input shapes, output clamped to [0, 1]).
SPAN architecture: Wan et al., "Swift Parameter-free Attention Network for Efficient Super-Resolution" (NTIRE 2024).

## Tools only (not shipped in the app)

`Tools/ml/` uses PyTorch, coremltools (BSD-3-Clause), spandrel (MIT), safetensors, Pillow and
huggingface_hub, installed into a local virtual environment.
