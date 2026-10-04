# Stage 9–15: pipeline, real-time, Dynamic Quality и производительность на M4

Машина: Apple M4 (10 ядер CPU, 16 GB), macOS 26.6. Таблицы собраны `Tools/ml/make_report.py` из
`Docs/data/*.json`. Benchmark запускается из меню **Quality → Benchmark… (⌘B)** или headless:

```bash
scripts/run_benchmarks.sh 5
```

## Pipeline

```text
Video file
  └─ AVPlayer ── VideoToolbox (аппаратный декодер) ── звук, A/V sync, seek
       └─ AVPlayerItemVideoOutput ── CVPixelBuffer (IOSurface; 8-bit: &8v0 без конвертации, 10-bit: x420)
            └─ FrameProvider.copyFrame(forHostTime: vsync + lookahead)          [render thread, CADisplayLink]
                 └─ CVMetalTextureCache → YCbCr→RGB (Metal, BT.601/709/2020, PQ/HLG → SDR)
                      ├─ Original: → present
                      ├─ Fast:     → артефакты (bilateral) → Lanczos-2 → CAS-резкость → present
                      └─ AI:       → тензор FP16 в IOSurface (Metal) → Core ML (ANE) → в наш IOSurface (outputBackings)
                                    → RGB (Metal) → коррекция низких частот → временная стабилизация → present
                                                                              [Metal, CAMetalLayer, BT.709]
```

### Что копируется между CPU и GPU/ANE

Ничего из пикселей. Кадр декодера живёт в IOSurface. Metal читает его через `CVMetalTextureCache`, пишет вход
модели прямо в IOSurface, который Core ML видит как `MLMultiArray(pixelBuffer:)`. Core ML пишет результат в наш
IOSurface через `MLPredictionOptions.outputBackings`. CPU только кодирует команды. Промежуточные текстуры
берутся из пула (`TexturePool`), буферы Core ML тоже (`PixelBufferPools`). В установившемся режиме видеопамять
на кадр не выделяется.

## Real-time: lookahead и backpressure

* **Lookahead.** В режиме AI кадр берётся из декодера заранее, на p95-задержку AI плюс два vsync
  (`AVPlayerItemVideoOutput` отдаёт будущие кадры), и показывается ровно в своё время. Поэтому изображение не
  отстаёт от звука, хотя inference занимает десятки миллисекунд. «A/V offset» в мониторе = фактический показ
  (`presentedTime`) минус плановое время.
* **Backpressure.** В обработке не больше двух кадров: два в полёте дают +50% пропускной способности за счёт
  перекрытия pre/post-обработки и inference (замер ниже). Пока AI занят, хранится только самый новый
  ожидающий кадр. Предыдущий отбрасывается и учитывается как «superseded». Очередь не растёт, задержка ограничена.
* **Порядок.** Кадр, обработанный позже, чем уже показан более новый, не показывается («finished too late»).
  После seek идёт новая «эпоха», и кадры старой эпохи выбрасываются.
* **Пауза.** Сразу показывается Metal-вариант, затем его заменяет AI-результат: на паузе ограничения по
  времени нет.

## Dynamic Quality

Лестница для режима **AI**: AI на полном разрешении источника → Fast (Metal) → Original. Правила:

1. AI переключается на Fast, если его фактическая пропускная способность больше секунды ниже 92% нужной
   (`номинальный fps × скорость воспроизведения`). Повторная попытка будет после паузы (10 s, затем дольше) или
   если нужный fps упал, например при замедлении воспроизведения.
2. Вход модели **никогда не уменьшается** ради скорости: замер в `02_MODEL_EVALUATION.md` показал, что
   уменьшение входа + AI хуже Lanczos. Real-time AI применяется к источникам до 720p, AI High Quality до 1080p.
3. Fast переключается на Original, если его GPU-время превышает 80% бюджета кадра.
4. Режим **AI High Quality** не откатывается по скорости (по ТЗ: «даже если не успевает за real-time»),
   а честно показывает «Not real time: N of M fps».

Причина текущего выбора всегда видна: строка статуса под таймлайном и Performance Monitor (⌘I).

## Замеры внутри плеера (zero-copy, тот же код, что при воспроизведении)

<!-- benchmarks:start -->

**Compute units (2 frames in flight, frames arriving at 60 fps)**

With 2 frames in flight the prediction wall time includes waiting for the other frame on the Neural Engine; pure inference time is in the 1-frame table. Peak MB accumulates across runs in one process.

| Model | Compute | Input → Output | Avg FPS | Inference med / p95 ms | Total ms | Dropped | Peak MB | CPU % | GPU % | Error |
|---|---|---|---|---|---|---|---|---|---|---|
| span-nomosuni-x2 | cpuAndNeuralEngine | 640×360 → 1280×720 | 45.8 | 32.9 / 40.0 | 36.0 | 72/301 | 100 | 1 | 9 |  |
| span-nomosuni-x2 | cpuAndNeuralEngine | 960×540 → 1920×1080 | 20.6 | 85.3 / 93.2 | 90.6 | 198/301 | 210 | 1 | 9 |  |
| span-nomosuni-x2 | cpuAndNeuralEngine | 1280×720 → 2560×1440 | 11.6 | 158.4 / 166.8 | 166.9 | 243/301 | 382 | 1 | 9 |  |
| span-nomosuni-x2 | all | 640×360 → 1280×720 | 45.6 | 32.3 / 40.0 | 35.7 | 73/301 | 264 | 1 | 9 |  |
| span-nomosuni-x2 | all | 960×540 → 1920×1080 | 20.4 | 84.9 / 93.6 | 90.4 | 199/301 | 308 | 1 | 9 |  |
| span-nomosuni-x2 | all | 1280×720 → 2560×1440 | 11.6 | 158.3 / 165.6 | 166.8 | 243/301 | 387 | 0 | 9 |  |
| span-nomosuni-x2 | cpuAndGPU | 640×360 → 1280×720 | 17.0 | 104.0 / 115.1 | 107.6 | 216/301 | 595 | 2 | 5 |  |
| span-nomosuni-x2 | cpuAndGPU | 960×540 → 1920×1080 | 7.6 | 242.6 / 258.9 | 251.2 | 263/301 | 1243 | 2 | 5 |  |
| span-nomosuni-x2 | cpuAndGPU | 1280×720 → 2560×1440 | 4.4 | 452.8 / 464.8 | 465.8 | 279/301 | 2046 | 2 | 5 |  |
| realesr-animevideov3-x2 | cpuAndNeuralEngine | 640×360 → 1280×720 | 26.8 | 64.1 / 72.6 | 67.4 | 167/301 | 274 | 1 | 5 |  |
| realesr-animevideov3-x2 | cpuAndNeuralEngine | 960×540 → 1920×1080 | 12.0 | 158.0 / 166.1 | 164.0 | 241/301 | 319 | 1 | 5 |  |
| realesr-animevideov3-x2 | cpuAndNeuralEngine | 1280×720 → 2560×1440 | 6.8 | 290.5 / 298.8 | 299.8 | 267/301 | 398 | 1 | 5 |  |
| realesr-animevideov3-x2 | all | 640×360 → 1280×720 | 27.0 | 63.3 / 70.6 | 66.8 | 166/301 | 274 | 1 | 5 |  |
| realesr-animevideov3-x2 | all | 960×540 → 1920×1080 | 12.4 | 154.3 / 161.2 | 160.0 | 239/301 | 319 | 1 | 5 |  |
| realesr-animevideov3-x2 | all | 1280×720 → 2560×1440 | 7.2 | 277.4 / 284.6 | 286.3 | 265/301 | 398 | 0 | 5 |  |
| realesr-animevideov3-x2 | cpuAndGPU | 640×360 → 1280×720 | 15.2 | 121.3 / 128.0 | 126.2 | 225/301 | 466 | 2 | 2 |  |
| realesr-animevideov3-x2 | cpuAndGPU | 960×540 → 1920×1080 | 6.8 | 274.1 / 285.6 | 276.8 | 267/301 | 704 | 2 | 3 |  |
| realesr-animevideov3-x2 | cpuAndGPU | 1280×720 → 2560×1440 | 4.0 | 505.2 / 507.7 | 510.2 | 281/301 | 1093 | 1 | 2 |  |
| realesr-general-x4v3-x2 | cpuAndNeuralEngine | 640×360 → 1280×720 | 14.6 | 128.9 / 136.1 | 132.3 | 228/301 | 276 | 1 | 3 |  |
| realesr-general-x4v3-x2 | cpuAndNeuralEngine | 960×540 → 1920×1080 | 6.8 | 295.8 / 302.8 | 301.3 | 267/301 | 320 | 0 | 3 |  |
| realesr-general-x4v3-x2 | cpuAndNeuralEngine | 1280×720 → 2560×1440 | 4.0 | 535.8 / 547.9 | 545.4 | 281/301 | 399 | 0 | 3 |  |
| realesr-general-x4v3-x2 | all | 640×360 → 1280×720 | 14.6 | 127.9 / 135.5 | 131.2 | 228/301 | 276 | 1 | 3 |  |
| realesr-general-x4v3-x2 | all | 960×540 → 1920×1080 | 6.8 | 296.1 / 305.4 | 302.0 | 267/301 | 321 | 0 | 3 |  |
| realesr-general-x4v3-x2 | all | 1280×720 → 2560×1440 | 4.0 | 535.4 / 548.6 | 545.3 | 281/301 | 400 | 0 | 3 |  |
| realesr-general-x4v3-x2 | cpuAndGPU | 640×360 → 1280×720 | 8.0 | 240.4 / 266.6 | 244.3 | 261/301 | 444 | 3 | 1 |  |
| realesr-general-x4v3-x2 | cpuAndGPU | 960×540 → 1920×1080 | 3.6 | 539.9 / 551.0 | 544.4 | 283/301 | 700 | 2 | 1 |  |
| realesr-general-x4v3-x2 | cpuAndGPU | 1280×720 → 2560×1440 | 2.4 | 971.3 / 977.5 | 976.0 | 289/301 | 876 | 1 | 1 |  |
| metal (Lanczos+CAS) | GPU | 640×360 → 1280×720 | 1327.6 | 0.5 / 0.5 | — | 0/6638 | 407 | 10 | 61 |  |
| metal (Lanczos+CAS) | GPU | 960×540 → 1920×1080 | 775.5 | 1.0 / 1.0 | — | 0/3878 | 407 | 6 | 79 |  |
| metal (Lanczos+CAS) | GPU | 1280×720 → 2560×1440 | 452.1 | 1.8 / 2.1 | — | 0/2261 | 407 | 4 | 83 |  |

**1 frame in flight (CPU+ANE)**

| Model | Compute | Input → Output | Avg FPS | Inference med / p95 ms | Total ms | Dropped | Peak MB | CPU % | GPU % | Error |
|---|---|---|---|---|---|---|---|---|---|---|
| span-nomosuni-x2 | cpuAndNeuralEngine | 640×360 → 1280×720 | 29.6 | 24.1 / 26.5 | 26.9 | 153/301 | 94 | 3 | 5 |  |
| span-nomosuni-x2 | cpuAndNeuralEngine | 960×540 → 1920×1080 | 15.2 | 50.1 / 51.0 | 54.9 | 225/301 | 188 | 1 | 5 |  |
| span-nomosuni-x2 | cpuAndNeuralEngine | 1280×720 → 2560×1440 | 10.2 | 86.7 / 87.2 | 94.0 | 250/301 | 334 | 1 | 6 |  |
| realesr-animevideov3-x2 | cpuAndNeuralEngine | 640×360 → 1280×720 | 20.2 | 37.8 / 39.9 | 41.1 | 200/301 | 307 | 1 | 3 |  |
| realesr-animevideov3-x2 | cpuAndNeuralEngine | 960×540 → 1920×1080 | 10.2 | 83.0 / 85.6 | 88.4 | 250/301 | 284 | 1 | 4 |  |
| realesr-animevideov3-x2 | cpuAndNeuralEngine | 1280×720 → 2560×1440 | 6.2 | 146.7 / 148.4 | 154.3 | 270/301 | 336 | 0 | 4 |  |
| realesr-general-x4v3-x2 | cpuAndNeuralEngine | 640×360 → 1280×720 | 12.2 | 70.3 / 71.2 | 73.7 | 240/301 | 307 | 1 | 2 |  |
| realesr-general-x4v3-x2 | cpuAndNeuralEngine | 960×540 → 1920×1080 | 6.0 | 155.3 / 162.9 | 160.3 | 271/301 | 284 | 0 | 2 |  |
| realesr-general-x4v3-x2 | cpuAndNeuralEngine | 1280×720 → 2560×1440 | 3.6 | 276.5 / 278.3 | 283.8 | 283/301 | 336 | 0 | 2 |  |

**CPU only**

| Model | Compute | Input → Output | Avg FPS | Inference med / p95 ms | Total ms | Dropped | Peak MB | CPU % | GPU % | Error |
|---|---|---|---|---|---|---|---|---|---|---|
| span-nomosuni-x2-flex | cpuOnly | 640×360 → 1280×720 | 8.2 | 110.3 / 128.6 | 113.1 | 260/301 | 268 | 219 | 2 |  |
| realesr-animevideov3-x2-flex | cpuOnly | 640×360 → 1280×720 | 8.6 | 105.6 / 112.2 | 108.9 | 258/301 | 191 | 99 | 2 |  |
<!-- benchmarks:end -->

## Оптимизации под M4 (Stage 15) и их измеренный эффект

<!-- optimizations:start -->
| Оптимизация | Эффект (измерено на этом M4) | Статус |
|---|---|---|
| **Два кадра в обработке** (перекрытие pre/post-обработки и inference) | SPAN: 360p 29.6 → 45.8 fps (+55%), 540p 15.2 → 20.6 (+36%), 720p 10.2 → 11.6 (+14%); AnimeVideo v3 360p 20.2 → 26.8; General v3 360p 12.2 → 14.6. Цена: +9 ms задержки на 360p (26.9 → 36.0 ms), её покрывает lookahead | принято |
| **Neural Engine** (CPU+ANE / ALL) вместо GPU | в 1.6–2.6 раза быстрее CPU+GPU и меньше памяти. Пиковая память процесса, SPAN 720p, каждый режим в отдельном процессе: CPU+ANE 249 MB против 1460 MB у CPU+GPU. GPU-путь Core ML выделяет большие промежуточные буферы. В таблицах выше пики накапливаются между прогонами одного процесса | принято (по умолчанию CPU+ANE) |
| **Zero-copy I/O Core ML** (`MLMultiArray(pixelBuffer:)` + `outputBackings`) | подготовка входа 0.07–0.3 ms GPU; inference в плеере совпадает с изолированным замером, то есть доставка кадра модели ничего не добавляет | принято |
| **Нативный выход декодера** (`&8v0`, lossless-compressed) для 8-bit H.264/HEVC | убирает полнокадровый проход конвертации AVFoundation; Metal читает его побайтно так же, как `420v` (`Tools/compare_native_format.swift`: 0 различий) | принято (Automatic) |
| **Размытие невязки и маска движения на низком разрешении** | постобработка AI: 5.2 → 1.2 ms GPU на выходе 720p, 12.3 → 5.7 ms на 1440p (замер в процессе разработки) | принято |
| **Пулы текстур и IOSurface-буферов** | в установившемся режиме видеопамять на кадр не выделяется | принято |
| **Render-поток и отрисовка только при изменении** | на паузе GPU не работает; кадры не ждут вёрстку SwiftUI на главном потоке | принято |
| **INT8 W8A8** (ускоренный int8-путь ANE на M4) | SPAN −9…−12% времени, AnimeVideo v3 от −2% до +2% (эффекта нет); качество −0.0…−0.8 dB PSNR (таблица ниже) | **не принято**: выигрыш мал по сравнению с потерей качества |
| **Фиксированные формы вместо перечисляемых** | 1–4% быстрее на ANE | **не принято**: не стоит 11 отдельных пакетов на модель |
| **Только CPU** | падает на перечисляемых формах (баг Core ML); с вариантами `*-flex` (RangeDim) работает: ~8 fps на 360p | поддержано для тестов; не для воспроизведения |
<!-- optimizations:end -->

## INT8 (W8A8): эксперимент для Neural Engine M4

<!-- int8:start -->
CPU+ANE, median ms (Python coremltools, subprocess per model). W8A8 = int8 weights + int8 activations, calibrated on natural photo crops.

| Model | Input | Enumerated FP16 (shipped) | Fixed FP16 | Fixed W8A8 | W8A8 time vs FP16 | W8A8 vs FP16 PSNR | GT PSNR FP16 | GT PSNR W8A8 |
|---|---|---|---|---|---|---|---|---|
| span-nomosuni | 640x360 | 24.0 | 23.0 | 20.3 | −12% | 40.4 | 30.16 | 29.32 |
| span-nomosuni | 960x540 | 53.1 | 51.8 | 47.1 | −9% | 41.6 | 30.40 | 30.37 |
| span-nomosuni | 1280x720 | 94.7 | 93.5 | 84.4 | −10% | 41.8 | 30.63 | 30.07 |
| realesr-animevideov3 | 640x360 | 40.1 | 39.5 | 38.7 | −2% | 44.0 | 31.86 | 31.70 |
| realesr-animevideov3 | 960x540 | 88.0 | 86.5 | 87.7 | +1% | 42.1 | 32.24 | 32.01 |
| realesr-animevideov3 | 1280x720 | 156.5 | 154.0 | 156.5 | +2% | 44.1 | 32.61 | 32.51 |
<!-- int8:end -->

## Что измеряется, а что нет

| Метрика | Источник |
|---|---|
| Video / Display / Processing FPS | счётчики кадров: из декодера, показанных (`presentedTime`), из AI |
| Frame time, inference, pre/post GPU | `MTLCommandBuffer.gpuStartTime/gpuEndTime`, время вызова `MLModel.prediction` |
| Latency, A/V offset | `CAMetalDrawable.presentedTime` минус время выборки или плановое время |
| Dropped | backpressure, опоздание, пропуски PTS у декодера |
| Memory | `task_vm_info.phys_footprint` (как в Xcode / Activity Monitor) |
| CPU | `getrusage`: время CPU процесса ÷ реальное время (100% = одно ядро) |
| GPU | доля времени, которую GPU выполнял **наши** command buffers; GPU-работа Core ML сюда не входит |
| Neural Engine | **загрузка недоступна через публичный API.** Показывается план `MLComputePlan` (какие операции назначены на ANE) и время inference |
| Thermal | `ProcessInfo.thermalState` |
