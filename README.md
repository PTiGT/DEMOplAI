# Video Enhancer

Нативный macOS-видеоплеер для Apple Silicon (MacBook M4) с AI Super Resolution.
Swift 6 · SwiftUI · AVFoundation · VideoToolbox · Metal · Core ML. Без Electron и веб-обёрток.

```text
Video → VideoToolbox (HW decode) → CVPixelBuffer (IOSurface) → Metal → Core ML (Neural Engine) → Metal → Display
```

## Документы

| | |
|---|---|
| [Docs/01_MODEL_SELECTION.md](Docs/01_MODEL_SELECTION.md) | Stage 0: сравнение моделей с Hugging Face и выбор |
| [Docs/02_MODEL_EVALUATION.md](Docs/02_MODEL_EVALUATION.md) | Stage 6–8: скачивание, конвертация, тест моделей отдельно от плеера (качество, мерцание, скорость) |
| [Docs/03_PERFORMANCE.md](Docs/03_PERFORMANCE.md) | Stage 9–15: pipeline, real-time, Dynamic Quality, benchmark, оптимизации под M4 |

## Статус этапов

| Stage | | |
|---|---|---|
| 1 | Видеоплеер: Open, Drag & Drop, Play/Pause, Seek, Volume, Mute, Fullscreen, Speed, Previous/Next | ✅ |
| 2 | `CVPixelBuffer` из видео (`AVPlayerItemVideoOutput`, IOSurface, без копий) | ✅ |
| 3 | Metal-рендер: `CAMetalLayer`, render-поток, YCbCr→RGB (601/709/2020, 8/10-bit, PQ/HLG→SDR) | ✅ |
| 4 | GPU upscale: Lanczos-2 с anti-ringing | ✅ |
| 5 | Enhancement: шумоподавление/артефакты сжатия, адаптивная резкость | ✅ |
| 6 | Модели с Hugging Face (Real-ESRGAN v3 ×2, SPAN) с фиксацией ревизий | ✅ |
| 7 | Тест моделей отдельно от плеера | ✅ |
| 8 | Конвертация в Core ML (FP16, enumerated shapes) | ✅ |
| 9 | Core ML в `VideoProcessingEngine` (zero-copy I/O) | ✅ |
| 10 | Real-time: lookahead, backpressure, порядок кадров, временная стабилизация | ✅ |
| 11 | Dynamic Quality | ✅ |
| 12 | Original vs Enhanced: Split View и Slider | ✅ |
| 13 | Performance Monitor | ✅ |
| 14 | Benchmark (в приложении и headless) | ✅ |
| 15 | Оптимизация под M4 | ✅ (см. `03_PERFORMANCE.md`) |

## Честно о real-time AI на M4

Самая быстрая подходящая модель (SPAN ×2) на Neural Engine M4 укладывается в real-time для источников
**до 360p при 24–30 fps**. Для 540p и 720p она даёт около 20 и 12 fps. Поэтому в режиме **AI** плеер:
- применяет AI там, где он успевает;
- иначе честно переходит на Metal Enhancement и пишет причину;
- на паузе всегда показывает полный AI-результат.

Режим **AI High Quality** обрабатывает кадры полным AI даже медленнее real-time. Точные цифры лежат в
`Docs/03_PERFORMANCE.md`.

## Режимы качества (меню Quality, ⌘1–⌘4)

| Режим | Что делает |
|---|---|
| Original | без обработки |
| Fast | Metal: шумоподавление/артефакты → Lanczos-2 до размера экрана → адаптивная резкость. Не AI |
| AI | Core ML-модель (по умолчанию SPAN ×2) на полном разрешении источника + коррекция низких частот + временная стабилизация. Dynamic Quality |
| AI High Quality | более тяжёлая модель (Real-ESRGAN General v3), ×4 при большом экране, без отката по скорости |

## Сборка и запуск

```bash
git clone https://github.com/PTiGT/PlayerAiupDEMO.git && cd PlayerAiupDEMO
```

Нужны Swift toolchain (Command Line Tools или Xcode 16+) и macOS 15+. Модели в `Models/` уже сконвертированы.
Скрипт сборки компилирует их в `.mlmodelc`, если есть Xcode, иначе приложение скомпилирует их при первом запуске.

```bash
scripts/build_app.sh
```

```bash
open build/VideoEnhancer.app
```

Headless benchmark:

```bash
scripts/run_benchmarks.sh 5
```

Пересоздать модели и отчёты (нужен Python 3.12, ~1 GB зависимостей):

```bash
/opt/homebrew/bin/python3.12 -m venv .venv && .venv/bin/pip install torch coremltools numpy safetensors spandrel pillow huggingface_hub
```

```bash
.venv/bin/python Tools/ml/download_models.py && .venv/bin/python Tools/ml/convert_to_coreml.py
```

### Тестовые ролики

```bash
swift Tools/make_test_video.swift TestMedia/pattern_720p60.mp4 1280 720 60 10
```

```bash
swift Tools/make_test_video.swift TestMedia/natural_360p24.mp4 640 360 24 20 h264 TestMedia/images/aerial_4k.png
```

Второй вариант делает панораму по натуральному фото (сначала экспортируйте системные обои через `sips`,
см. `Docs/02_MODEL_EVALUATION.md`), со «стриминговым» битрейтом.

## Клавиши

| Клавиша | Действие |
|---|---|
| Space | Play / Pause |
| ← / → (Shift: ±30 s) | −5 / +5 s |
| ⌘← / ⌘→ | Previous / Next |
| ↑ / ↓ · M | Громкость · Mute |
| [ / ] | Медленнее / быстрее |
| F | Full Screen |
| ⌘1 … ⌘4 | Original / Fast / AI / AI High Quality |
| ⌘K | Сравнение: Off → Split → Slider |
| ⌘I | Performance Monitor |
| ⌘B | Benchmark |

## Структура

```text
Sources/VideoEnhancer/
  App/          EntryPoint (+ headless benchmark), VideoEnhancerApp, PlayerCommands, AppResources
  UI/           PlayerView, PlayerControls, SettingsView (Quality Boost + Advanced), PerformanceView,
                ComparisonView, BenchmarkView, MetalVideoView
  Player/       VideoPlayer (сборка pipeline), PlaybackController, FrameProvider, Playlist
  Video/        VideoFrame, ColorInfo
  Processing/   VideoProcessingEngine, SuperResolutionProcessor, EnhancementPipeline, Upscaler,
                DynamicQualityController, EnhancementController, ProcessedFrame
  Metal/        MetalContext, MetalRenderer, MetalTextureCache, TexturePool, YCbCrConverter, RenderThread,
                Shaders/ (Common, Convert, Enhance, ML, Render — компилируются в runtime)
  ML/           SuperResolutionModel (протокол), CoreMLSuperResolution, ModelManager
  Performance/  PerformanceMonitor, FPSCounter, FrameMetrics, SystemMetrics, BenchmarkRunner
  Models/       VideoInfo, ProcessingConfiguration
Tools/          make_test_video.swift, check_shaders.swift, compare_native_format.swift, ml/*.py
Models/         *.mlpackage (+ source/ с исходными весами и manifest.json)
Docs/           решения, замеры (data/*.json), картинки
```

## Расширяемость

- **`SuperResolutionModel`** — протокол бэкенда. Он объявляет раскладку входа и выхода и `temporalFrames`.
  Новая модель (в том числе video SR с несколькими кадрами) подключается без изменений в pipeline.
  `SRFrameContext` передаёт последовательность и разрывы, нужные temporal-моделям.
- **Новые модели** находит сам `ModelManager`: любые `.mlpackage`, сконвертированные
  `Tools/ml/convert_to_coreml.py`, в `Models/` или `~/Library/Application Support/VideoEnhancer/Models`.
- **Frame Interpolation, HDR, AI denoise/deblock** встраиваются как дополнительные шаги в
  `VideoProcessingEngine.process` с тем же lookahead и backpressure.
- **Системные ML-апскейлеры macOS 26** (`VTSuperResolutionScaler` с temporal VSR и
  `VTLowLatencySuperResolutionScaler`) проверены на этом M4 (`Docs/01_MODEL_SELECTION.md`). Это следующий
  кандидат на бэкенд.

## Лицензии моделей

Real-ESRGAN (Xintao Wang et al.): BSD-3-Clause. SPAN 2xNomosUni_span_multijpg (Philip Hofmann): CC-BY-4.0.
Источники и ревизии перечислены в `Models/source/manifest.json` и в метаданных каждого `.mlpackage`.
Тексты лицензий и атрибуции: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
