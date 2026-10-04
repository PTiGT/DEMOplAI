# Выбор Super Resolution модели для VideoEnhancer (MacBook M4)

Дата анализа: 2026-10-04. Целевая машина: Apple M4, 16 GB, macOS 26.6.

Документ разделяет **факты** (проверены на Hugging Face, в заголовках SDK или запуском на этом Mac)
и **оценки** (расчёт по числу операций, помечены «≈ оценка»). Реальные цифры скорости появятся только
в Stage 7 (тест модели отдельно от плеера) — до этого никакой FPS не считается подтверждённым.

---

## 1. Главное ограничение: вычислительный бюджет

Полностью свёрточные SR-сети (SRVGGNetCompact, SPAN) работают на разрешении **входа**, поэтому
стоимость кадра ≈ `параметры × пиксели входа` (каждый вес 3×3 используется ~1 раз на пиксель).

| Вход | Пикселей | Бюджет кадра при 60 fps | при 30 fps | при 24 fps |
|---|---|---|---|---|
| 640×360 | 230 400 | 16.7 ms | 33.3 ms | 41.7 ms |
| 960×540 | 518 400 | 16.7 ms | 33.3 ms | 41.7 ms |
| 1280×720 | 921 600 | 16.7 ms | 33.3 ms | 41.7 ms |

Бюджет общий: декодирование, Metal pre/post-processing и отрисовка тоже в него входят.

## 2. Кандидаты

| # | Модель (HF) | Архитектура | Scale | Параметры / файл | Лицензия | Стоимость кадра 720p (≈ оценка) | Real-time на M4 |
|---|---|---|---|---|---|---|---|
| 1 | **Real-ESRGAN `realesr-animevideov3`** — [mlx-community/Real-ESRGAN-animevideov3](https://huggingface.co/mlx-community/Real-ESRGAN-animevideov3) (веса из официального релиза xinntao) | SRVGGNetCompact: 64 канала, 16 conv, PReLU, PixelShuffle | ×4 (используется и как ×2/×3 через resize) | **0.62 M**, 1.25 MB fp16 | **BSD-3-Clause** | ≈ 0.57 TMAC | 720p — нет; ≤540p вход — вероятно (проверить) |
| 2 | **Real-ESRGAN `realesr-general-x4v3`** — [qualcomm/Real-ESRGAN-General-x4v3](https://huggingface.co/qualcomm/Real-ESRGAN-General-x4v3), [mlx-community/Real-ESRGAN-general-x4v3](https://huggingface.co/mlx-community/Real-ESRGAN-general-x4v3) | SRVGGNetCompact: 64 канала, 32 conv, PReLU + denoise-вариант (wdn) | ×4 | **1.21 M**, 4.65 MB fp32 | **BSD-3-Clause** | ≈ 1.1 TMAC | Нет (кроме ~360p входа) |
| 3 | **SPAN ×2** — [Phips/2xNomosUni_span_multijpg](https://huggingface.co/Phips/2xNomosUni_span_multijpg) | SPAN (победитель NTIRE 2024 Efficient SR): conv + attention без параметров (sigmoid·mul) | **×2 нативно** | ≈ 0.4 M (по статье SPAN; уточнить при загрузке) | **CC-BY-4.0** (нужна атрибуция) | ≈ 0.37 TMAC | ≤540p — вероятно; 720p — под вопросом |
| 4 | Real-ESRGAN `x2plus` / `x4plus` — [Comfy-Org/Real-ESRGAN_repackaged](https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged), [qualcomm/Real-ESRGAN-x4plus](https://huggingface.co/qualcomm/Real-ESRGAN-x4plus) | RRDBNet, 23 RRDB блока | ×2 / ×4 | 16.7 M, ~64 MB | BSD-3-Clause | ≈ 4 TMAC (x2plus) | Нет. Только офлайн / «AI High Quality» |
| 5 | Swin2SR — [caidas/swin2SR-lightweight-x2-64](https://huggingface.co/caidas/swin2SR-lightweight-x2-64), [caidas/swin2SR-classical-sr-x2-64](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64) | Swin Transformer V2 (оконное внимание) | ×2 / ×4 | ~1 M (lightweight) … ~12 M | Apache-2.0 | ≥ 1 TMAC + attention | Нет |
| — | HAT (XPixelGroup) | гибрид CNN + attention | ×2 / ×4 | ~20.8 M | Apache-2.0 | ≈ 19 TMAC | Нет, только как эталон качества офлайн |

### Пояснения по каждой модели

**1. realesr-animevideov3.** Самая лёгкая официальная модель Real-ESRGAN, авторы выпустили её именно
для видео (аниме). Все операции (Conv2d 3×3, PReLU, PixelShuffle, сложение с bilinear/nearest-upsample
входа) нативно поддерживаются Core ML и хорошо подходят для Neural Engine. Минус: обучена на аниме,
поэтому на живом видео может давать «рисованную» текстуру. Это **image SR**: соседние кадры не используются.

**2. realesr-general-x4v3.** Та же архитектура, но в 2 раза глубже, обучена на реальных деградациях
(шум, размытие, JPEG). Это правильная модель для фильмов. Есть внешняя точка замера: Qualcomm AI Hub —
вход 128×128 → 3.5 ms на NPU Snapdragon 8 Elite (float). Линейная экстраполяция на 720p (в 56 раз больше
пикселей) даёт ≈ 200 ms на кадр на том NPU. Это ориентир, а не замер на M4.

**3. SPAN ×2.** Нативный ×2 (не надо считать ×4 и потом уменьшать), обучена с JPEG-деградацией до качества 40.
Это близко к артефактам сжатого видео. Чуть легче animevideov3. Минусы: модель от сообщества,
CC-BY-4.0 требует атрибуции, шире разброс поведения на разном контенте.

**4. RRDBNet x2plus/x4plus.** Лучшее качество в семействе Real-ESRGAN, конвертируется в Core ML, но примерно
в 7–8 раз тяжелее compact-моделей. Официальная Core ML-конверсия на HF ([mlboydaisuke/Real-ESRGAN-x4-CoreML](https://huggingface.co/mlboydaisuke/Real-ESRGAN-x4-CoreML))
требует ~500 MB RAM при тайле 512×512, и её автор отдельно описывает проблемы FP16 overflow и ограничения буферов ANE.
Для real-time не годится.

**5. Swin2SR / HAT.** Высокое PSNR на синтетической (bicubic) деградации. Window-attention
(window partition, roll, softmax на больших тензорах) плохо ложится на Neural Engine, плюс трудности с переменным
размером входа. Classical-варианты не обучены на сжатом видео. Для плеера не подходят.

### Video Super Resolution (несколько кадров)

Настоящие VSR-модели (BasicVSR++, RealBasicVSR, EDVR) используют deformable convolution и/или warping по
оптическому потоку. Deformable conv не имеет нативного op в Core ML, и все они на порядок тяжелее compact-моделей.
Готового Core ML-пригодного VSR на Hugging Face я не нашёл.

**Важная находка (проверено на этом M4):** в macOS 26 VideoToolbox содержит встроенные ML-апскейлеры Apple
(`VTFrameProcessor`). Это **не** модели с Hugging Face, а закрытые системные модели, но это публичный API:

| API | Что делает | Результат проверки на этом M4 |
|---|---|---|
| `VTSuperResolutionScalerConfiguration` | **Temporal VSR**: принимает текущий кадр, предыдущий кадр, предыдущий выход и optical flow | supported = true, scale = **[4]** только; вход видео до 1920×1080 (по заголовку SDK); конфигурация 720p ×2 → `nil`; модель докачивается системой |
| `VTLowLatencySuperResolutionScalerConfiguration` | Покадровый SR с низкой задержкой | supported = true, max вход **960×960**, ×2 для 640×360 / 854×480 / 960×540; **для 1280×720 — не поддерживается** |

Поэтому архитектура плеера будет иметь протокол `SuperResolutionModel` с несколькими реализациями:
`CoreMLSuperResolution` (модели с HF), `VTSuperResolution` (системный temporal VSR), `MetalUpscaler` (не-AI).

## 3. Выбор для M4

**Основная модель: семейство Real-ESRGAN v3 compact (SRVGGNetCompact)**
- `realesr-animevideov3` — первая модель для конвертации и real-time режима **AI**;
- `realesr-general-x4v3` — для живого видео и режима **AI High Quality**.

**Контрольный кандидат для Stage 7:** SPAN ×2 (`Phips/2xNomosUni_span_multijpg`).

Почему так:
1. **Лицензия.** BSD-3-Clause, официальные веса авторов. Требуется только сохранить copyright-notice (сделаем экран Licenses).
2. **Core ML.** Только Conv/PReLU/PixelShuffle/Add. Нет attention, deformable conv и динамических форм. Хорошие шансы
   на полное размещение на Neural Engine (проверим через `MLComputePlan`).
3. **Размер и память.** 1.2–4.7 MB весов. Пиковая память определяется активациями
   (64 канала × пиксели × fp16: ≈ 118 MB на один промежуточный тензор при 720p), а не весами.
4. **Одна модель, несколько уровней нагрузки.** Стоимость линейна по пикселям входа, поэтому ступени качества делаются
   масштабом входа: 720p → 540p → 360p на вход сети, затем ×4 и Metal-downscale до размера экрана.
   Модель одна и та же, загружена один раз, а ступени переключаются без перезагрузки (Core ML EnumeratedShapes).
5. **Честная оценка real-time (≈ оценка, проверяется в Stage 7).** Полный 720p-вход ×4 даже для самой лёгкой модели
   (≈ 1.1 TFLOP/кадр) почти наверняка **не даст 60 fps** на M4. Реалистичная цель: AI для 24/30 fps контента при входе
   ≤540p и автоматический откат на Metal-улучшение для 60 fps.

**Режим compute unit не выбирается наугад.** В Stage 7 и 14 модель прогоняется во всех режимах
`MLComputeUnits` (`.cpuOnly`, `.cpuAndGPU`, `.cpuAndNeuralEngine`, `.all`). Распределение операций по устройствам
читается через публичный `MLComputePlan`. Процент загрузки ANE публичным API получить **нельзя**,
и плеер не будет его показывать.

## 4. Temporal flickering

Real-ESRGAN — GAN-модель, она дорисовывает текстуры, которых не было во входе. Покадровая обработка может приводить
к «кипению» мелких деталей. План:
1. Stage 10: temporal stabilization в Metal (смешивание с предыдущим выходом, где кадр почти не изменился,
   со сбросом на сменах сцены).
2. Архитектура `SuperResolutionModel` принимает **контекст кадров** (текущий + история), а не один буфер. Так можно
   подключить настоящий VSR (`VTSuperResolutionScaler` или будущую Core ML VSR-модель) без переделки pipeline.

## 5. Что требуется от окружения

Проверено на этом Mac:
- **Xcode 26.4.1 установлен** (`/Applications/Xcode.app`), но активным выбран Command Line Tools
  (`xcode-select -p` → `/Library/Developer/CommandLineTools`). Скрипты проекта не зависят от этого: сборка идёт через SwiftPM
  и `scripts/build_app.sh`. Чтобы Xcode-инструменты работали из терминала, переключите активный toolchain:
  `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`
  (или передавайте `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` в отдельные команды).
- `coremlcompiler` из Xcode работает (версия 3520.5.1), значит `.mlpackage` → `.mlmodelc` можно компилировать офлайн.
- Офлайн-компилятор `metal` требует отдельного компонента Xcode 26:
  `xcodebuild -downloadComponent MetalToolchain`. До его установки шейдеры компилируются в runtime через
  `MTLDevice.makeLibrary(source:)`. Для Stage 3–5 этого достаточно.
- **Instruments** (шаблоны Core ML и Metal System Trace) доступен вместе с Xcode. Это единственный честный способ увидеть
  на таймлайне, где реально выполнялась модель (ANE / GPU / CPU). Понадобится в Stage 7 и 15.
- Для конвертации (Stage 8) нужен Python ≥ 3.10 с `torch` и `coremltools`. Системный Python 3.9.6 не подойдёт,
  поставим `python@3.12` через Homebrew в отдельный venv.

---

## Итог после замеров (Stage 7–15)

Первоначальный выбор пересмотрен по данным из `02_MODEL_EVALUATION.md` и `03_PERFORMANCE.md`:

| Роль | Модель | Почему |
|---|---|---|
| **AI** (real-time, по умолчанию) | **SPAN ×2** (`Phips/2xNomosUni_span_multijpg`) | самая быстрая на ANE M4: 45.8 fps на 360p, 20.6 на 540p, 11.6 на 720p при 2 кадрах в обработке. Обучена на JPEG-артефактах. Её систематическое затемнение (−5…−7/255) убирает коррекция низких частот |
| **AI High Quality** | **Real-ESRGAN General v3** | лучшая точность среди AI-моделей до коррекции и натуральный вид на живом видео; медленнее всех (14.6 fps на 360p) |
| Аниме | Real-ESRGAN AnimeVideo v3 | «рисованный» вид, который на живом видео выглядит как обводка; для аниме по назначению |

Планировавшиеся альтернативы:
- **Квантизация INT8 и фиксированные формы.** Проверены и отклонены: выигрыш ≤ 12% (`03_PERFORMANCE.md`).
- **Системные апскейлеры VideoToolbox.** Остаются следующим кандидатом на бэкенд: на этом M4 low-latency SR принимает вход до 960×540, temporal SR только ×4.
