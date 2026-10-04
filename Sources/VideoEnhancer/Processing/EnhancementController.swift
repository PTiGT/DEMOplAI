import Foundation
import Observation

/// Main-thread bridge between the UI and the processing engine: owns the settings, picks and
/// prepares models for the current video, and publishes live performance numbers.
@MainActor
@Observable
final class EnhancementController {
    let modelManager = ModelManager()

    var configuration: ProcessingConfiguration {
        didSet {
            guard configuration != oldValue else { return }
            configuration.save()
            engine?.update(configuration: configuration)
            if configuration.mode != oldValue.mode || configuration.aiModelFamily != oldValue.aiModelFamily
                || configuration.hqModelFamily != oldValue.hqModelFamily || configuration.computeUnits != oldValue.computeUnits {
                prepareModels()
            }
        }
    }

    var comparison = ComparisonState() {
        didSet { renderer?.setComparison(comparison) }
    }

    enum ComparisonMode: String, CaseIterable, Identifiable {
        case off = "Off", split = "Split View", slider = "Slider"
        var id: String { rawValue }
    }

    var comparisonMode: ComparisonMode = .off {
        didSet {
            comparison.enabled = comparisonMode != .off
            if comparisonMode == .split { comparison.position = 0.5 }
        }
    }

    private(set) var snapshot = PerformanceSnapshot()
    var showBenchmark = false
    let benchmark = BenchmarkController()
    /// What the AI side is doing, in words (shown as "AI Model: …").
    private(set) var modelStatus = "Not loaded"
    private(set) var activeRealtimeModel: SuperResolutionModelInfo?
    private(set) var activeHQModel: SuperResolutionModelInfo?

    @ObservationIgnored private let engine: VideoProcessingEngine?
    @ObservationIgnored private let renderer: MetalRenderer?
    @ObservationIgnored private var videoInfo: VideoInfo?
    @ObservationIgnored private var prepareTask: Task<Void, Never>?
    @ObservationIgnored private var pollTask: Task<Void, Never>?

    init(engine: VideoProcessingEngine?, renderer: MetalRenderer?) {
        self.engine = engine
        self.renderer = renderer
        configuration = ProcessingConfiguration.load()
        engine?.update(configuration: configuration)
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                self?.poll()
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
        Task { [weak self] in
            await self?.modelManager.discover()
            self?.prepareModels()
        }
    }

    var aiBoostOn: Bool { snapshot.pathLabel.hasPrefix("AI") }

    func videoDidChange(_ info: VideoInfo?) {
        videoInfo = info
        prepareModels()
    }

    private func poll() {
        guard let engine else { return }
        snapshot = engine.monitor.snapshot()
    }

    // MARK: - Models

    /// Loads (once) and warms up the models the current mode needs for the current video.
    func prepareModels() {
        prepareTask?.cancel()
        guard let engine else { return }
        let config = configuration
        guard config.mode.usesAI else {
            engine.setModels(.init())
            modelStatus = activeRealtimeModel == nil ? "Not loaded" : "Loaded, not in use (\(config.mode.title) mode)"
            return
        }
        let family = config.mode == .aiHighQuality ? config.hqModelFamily : config.aiModelFamily

        prepareTask = Task { [weak self] in
            guard let self else { return }
            if modelManager.isDiscovering || modelManager.entries.isEmpty { await modelManager.discover() }
            guard let entry = chooseVariant(family: family, highQuality: config.mode == .aiHighQuality) else {
                let message = modelManager.entries.isEmpty
                    ? "No AI model installed (Models/*.mlpackage)"
                    : config.computeUnits == .cpuOnly ? "CPU only needs a -flex model variant of \(family)"
                    : "Model \(family) not found"
                engine.setModels(.init(problem: message))
                modelStatus = "Not loaded — \(message)"
                return
            }
            engine.setModels(.init())
            modelStatus = "Loading \(entry.info.displayName) ×\(entry.info.scale) (\(config.computeUnits.shortTitle))…"
            do {
                let model = try await modelManager.load(id: entry.id, units: config.computeUnits)
                if Task.isCancelled { return }
                if let info = videoInfo,
                   let size = SuperResolutionProcessor.inputSize(for: PixelSize(width: info.width, height: info.height), sizes: entry.info.inputSizes),
                   modelManager.entry(id: entry.id)?.warmUpSeconds[size] == nil {
                    modelStatus = "Preparing \(entry.info.displayName) for \(size) (first run compiles for the device)…"
                    await modelManager.warmUp(model, sizes: [size])
                    if Task.isCancelled { return }
                }
                if config.mode == .aiHighQuality {
                    engine.setModels(.init(highQuality: model))
                    activeHQModel = model.info
                } else {
                    engine.setModels(.init(realtime: model))
                    activeRealtimeModel = model.info
                }
                modelStatus = "\(model.info.displayName) ×\(model.info.scale) · \(config.computeUnits.shortTitle)"
                _ = await modelManager.computePlan(id: entry.id, units: config.computeUnits)
            } catch {
                if Task.isCancelled { return }
                engine.setModels(.init(problem: "AI model failed to load: \(error.localizedDescription)"))
                modelStatus = "Failed — \(error.localizedDescription)"
            }
        }
    }

    /// ×2 for real time (less output bandwidth); for High Quality ×4 when ×2 would leave the picture
    /// well below the screen resolution.
    private func chooseVariant(family: String, highQuality: Bool) -> ModelManager.Entry? {
        let variants = modelManager.variants(family: family)
        guard !variants.isEmpty else { return nil }
        var preferred = 2
        if highQuality, let info = videoInfo, let drawable = renderer?.currentDrawableSize, drawable.height > 0 {
            let fit = AspectFit.rect(content: CGSize(width: info.width, height: info.height), in: drawable)
            if Double(info.height * 2) < Double(fit.height) * 0.85 { preferred = 4 }
        }
        return modelManager.variant(family: family, preferredScale: preferred, units: configuration.computeUnits)
    }

    func unloadModels() {
        prepareTask?.cancel()
        engine?.setModels(.init())
        modelManager.unloadAll()
        activeRealtimeModel = nil
        activeHQModel = nil
        modelStatus = "Not loaded"
    }

    func reloadModels() {
        unloadModels()
        Task {
            await modelManager.discover()
            prepareModels()
        }
    }
}
