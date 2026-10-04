import CoreGraphics
import CoreMedia
import Foundation
import Metal
import QuartzCore

/// Turns decoded frames into `ProcessedFrame`s for the renderer. Called on the render thread once per vsync.
///
/// Real-time scheduling for AI:
///  * Lookahead — frames are pulled `lookahead` seconds before they are due (AVPlayerItemVideoOutput
///    can vend future frames), processed asynchronously and shown at their due time, so the picture
///    stays in sync with the audio even though inference takes tens of milliseconds.
///  * Backpressure — at most `SuperResolutionProcessor.maxInFlight` frames are processing; of the
///    frames that arrive meanwhile only the newest is kept (older ones are counted as dropped).
///    Nothing queues up, so latency stays bounded.
///  * Ready frames are shown in presentation order; one that finishes after a newer frame was shown is dropped.
final class VideoProcessingEngine: @unchecked Sendable {
    struct Models: Sendable {
        var realtime: (any SuperResolutionModel)?
        var highQuality: (any SuperResolutionModel)?
        var problem: String?
    }

    let context: MetalContext
    let pool: TexturePool
    let monitor: PerformanceMonitor
    private let frameProvider: FrameProvider
    private let converter: YCbCrConverter
    private let enhancement: EnhancementPipeline
    private let processor: SuperResolutionProcessor

    // Settings (set from the main thread).
    private let lock = NSLock()
    private var configuration = ProcessingConfiguration()
    private var generation = 0
    private var targetSize: CGSize = .zero
    private var models = Models()

    // Render-thread state.
    private var lastSource: SourceFrame?
    private var lastGeneration = -1
    private var lastTargetSize: CGSize = .zero
    private var epoch = 0
    private var pending: SourceFrame?
    private var lastShown: (sequence: UInt64, epoch: Int, isAI: Bool)?
    private var dynamic = DynamicQualityController()
    private var lastEvaluation: CFTimeInterval = 0
    private var aiActiveSince: CFTimeInterval?
    private var lookahead: CFTimeInterval = 0

    // Written by AI completion handlers.
    private let readyLock = NSLock()
    private var ready: [ProcessedFrame] = []

    init(context: MetalContext, frameProvider: FrameProvider, monitor: PerformanceMonitor) {
        self.context = context
        self.frameProvider = frameProvider
        self.monitor = monitor
        pool = TexturePool(device: context.device)
        converter = YCbCrConverter(context: context, pool: pool)
        enhancement = EnhancementPipeline(context: context, pool: pool)
        processor = SuperResolutionProcessor(context: context, pool: pool, enhancement: enhancement, monitor: monitor)
    }

    // MARK: - Settings (any thread)

    func update(configuration newValue: ProcessingConfiguration) {
        lock.withLock {
            guard configuration != newValue else { return }
            // A new mode or model deserves a fresh attempt, not the previous fallback's cooldown.
            if newValue.mode != configuration.mode || newValue.aiModelFamily != configuration.aiModelFamily
                || newValue.hqModelFamily != configuration.hqModelFamily || newValue.computeUnits != configuration.computeUnits {
                dynamic.reset()
            }
            configuration = newValue
            generation += 1
        }
    }

    /// New video: forget fallbacks and cooldowns from the previous one.
    func resetDynamicQuality() {
        lock.withLock { dynamic.reset() }
    }

    func setModels(_ newValue: Models) {
        lock.withLock {
            let changed = newValue.realtime?.info.id != models.realtime?.info.id
                || newValue.highQuality?.info.id != models.highQuality?.info.id
                || newValue.problem != models.problem
            models = newValue
            if changed { generation += 1 }
        }
        processor.resetHistory()
    }

    /// Drawable size in pixels; upscalers derive their output size from it.
    func setTargetSize(_ size: CGSize) {
        lock.withLock { targetSize = size }
    }

    var currentDecision: DynamicQualityController.Decision { lock.withLock { dynamic.decision } }

    // MARK: - Render thread

    /// Pulls the frame due at this vsync (if new) and encodes its processing into `commandBuffer`.
    func frame(forVsync targetTime: CFTimeInterval, interval: CFTimeInterval, commandBuffer: MTLCommandBuffer) -> ProcessedFrame? {
        let (config, gen, target, models) = lock.withLock { (configuration, generation, targetSize, self.models) }
        let rate = frameProvider.playbackRate(atHostTime: targetTime)
        let playing = abs(rate) > 0.01

        evaluateDynamicQuality(config: config, models: models, playing: playing, rate: rate, interval: interval, now: targetTime)
        let decision = lock.withLock { dynamic.decision }
        let aiModel = config.mode == .aiHighQuality ? models.highQuality : models.realtime
        let useAI = decision.path == .ai && aiModel != nil

        // Lookahead covers the AI latency while playing; 0 otherwise (and when paused: show at once).
        let desiredLookahead: CFTimeInterval = useAI && playing
            ? aiLookahead(interval: interval, cap: config.mode == .aiHighQuality ? 0.5 : 0.35) : 0
        lookahead = desiredLookahead
        monitor.updateContext { $0.lookaheadMs = desiredLookahead * 1000 }

        var immediate: ProcessedFrame?
        if let decoded = frameProvider.copyFrame(forHostTime: targetTime + lookahead) {
            if decoded.isDiscontinuity { resetTemporalState() }
            monitor.recordPulled(at: targetTime)
            do {
                let rgb = try converter.convert(decoded, into: commandBuffer)
                let source = SourceFrame(rgb: rgb, sequence: decoded.sequenceNumber, presentationTime: decoded.presentationTime,
                                         pulledAt: targetTime, dueAt: targetTime + lookahead, rotation: decoded.rotation,
                                         epoch: epoch, nominalFrameRate: decoded.nominalFrameRate)
                lastSource = source
                lastGeneration = gen
                lastTargetSize = target
                immediate = process(source, config: config, decision: decision, aiModel: useAI ? aiModel : nil,
                                    playing: playing, target: target, commandBuffer: commandBuffer)
            } catch {
                frameProvider.reportUnusableFrame(error)
            }
        } else if let source = lastSource, gen != lastGeneration || target != lastTargetSize {
            // Settings or window size changed without a new frame (e.g. paused): re-process the last one.
            lastGeneration = gen
            lastTargetSize = target
            immediate = process(source.redue(at: targetTime), config: config, decision: decision,
                                aiModel: useAI ? aiModel : nil, playing: false, target: target, commandBuffer: commandBuffer)
        }

        let due = takeDueFrame(before: targetTime + interval * 0.5)
        return choose(immediate: immediate, due: due)
    }

    private func process(_ source: SourceFrame, config: ProcessingConfiguration, decision: DynamicQualityController.Decision,
                         aiModel: (any SuperResolutionModel)?, playing: Bool, target: CGSize,
                         commandBuffer: MTLCommandBuffer) -> ProcessedFrame? {
        updateSizes(source: source, aiModel: aiModel, target: target)

        guard let aiModel else {
            if decision.path == .original { return source.frame(enhanced: nil, path: .original) }
            let enhanced = enhancement.encode(source.rgb, targetSize: target, rotation: source.rotation,
                                              settings: config.enhancement, into: commandBuffer)
            return source.frame(enhanced: enhanced, path: .metal)
        }

        if aiActiveSince == nil { aiActiveSince = CACurrentMediaTime() }
        submitOrPend(source, model: aiModel, config: config)
        if !playing {
            // Paused: show the Metal result now; the AI result replaces it when ready.
            let enhanced = enhancement.encode(source.rgb, targetSize: target, rotation: source.rotation,
                                              settings: config.enhancement, into: commandBuffer)
            return source.frame(enhanced: enhanced, path: .metal)
        }
        return nil
    }

    private func submitOrPend(_ source: SourceFrame, model: any SuperResolutionModel, config: ProcessingConfiguration) {
        let settings = AIPostSettings(consistency: config.consistencyCorrection, stabilization: config.temporalStabilization,
                                      sharpen: config.aiSharpen)
        let accepted = processor.submit(source, model: model, settings: settings) { [weak self] result in
            self?.aiFinished(result, model: model, config: config)
        }
        if !accepted {
            if pending != nil { monitor.recordDrop(.superseded) }
            pending = source
        }
    }

    private func aiFinished(_ result: Result<ProcessedFrame, Error>, model: any SuperResolutionModel, config: ProcessingConfiguration) {
        switch result {
        case .success(let frame):
            readyLock.withLock { ready.append(frame) }
        case .failure(let error):
            monitor.recordDrop(.failed)
            lock.withLock { models.problem = "AI failed: \(error.localizedDescription)" }
        }
        // Start the newest pending frame, if any. Runs on the render thread's next vsync otherwise;
        // starting here shaves up to one vsync off the latency.
        RenderThread.shared.perform { [weak self] in
            guard let self, let next = self.pending else { return }
            self.pending = nil
            self.submitOrPend(next, model: model, config: config)
        }
    }

    /// The newest finished AI frame that is due; older due frames are dropped.
    private func takeDueFrame(before deadline: CFTimeInterval) -> ProcessedFrame? {
        let due: [ProcessedFrame] = readyLock.withLock {
            let currentEpoch = epoch
            ready.removeAll { $0.epoch != currentEpoch }
            let due = ready.filter { $0.dueAt <= deadline }
            ready.removeAll { $0.dueAt <= deadline }
            return due
        }
        guard let newest = due.max(by: { $0.sequence < $1.sequence }) else { return nil }
        monitor.recordDrop(.late, count: due.count - 1)
        return newest
    }

    private func choose(immediate: ProcessedFrame?, due: ProcessedFrame?) -> ProcessedFrame? {
        var candidate = immediate
        if let due {
            let newerThanShown = lastShown.map { due.sequence > $0.sequence || (due.sequence == $0.sequence && !$0.isAI) } ?? true
            if newerThanShown, candidate == nil || due.sequence >= candidate!.sequence {
                candidate = due
            } else if !newerThanShown {
                monitor.recordDrop(.late)
            }
        }
        if let candidate {
            if case .ai = candidate.path {
                lastShown = (candidate.sequence, epoch, true)
            } else {
                lastShown = (candidate.sequence, epoch, false)
            }
        }
        return candidate
    }

    private func resetTemporalState() {
        epoch += 1
        pending = nil
        lastShown = nil
        readyLock.withLock { ready.removeAll() }
        processor.resetHistory()
    }

    private func aiLookahead(interval: CFTimeInterval, cap: CFTimeInterval) -> CFTimeInterval {
        let stats = monitor.aiStatistics
        guard let p95 = stats.latencyP95 else { return 0.12 }
        // Cover the slow end of the latency distribution plus two vsyncs of scheduling slack. The decoder
        // only works a limited distance ahead, hence the cap.
        return min(cap, max(0.03, p95 / 1000 + interval * 2))
    }

    private func updateSizes(source: SourceFrame, aiModel: (any SuperResolutionModel)?, target: CGSize) {
        let aiInput = aiModel.flatMap { SuperResolutionProcessor.inputSize(for: source.size, sizes: $0.info.inputSizes) }
        let output: PixelSize? = if let aiInput, let aiModel {
            aiInput.scaled(by: aiModel.info.scale)
        } else {
            {
                let size = MetalUpscaler.outputSize(source: CGSize(width: source.rgb.width, height: source.rgb.height),
                                                    target: target, rotation: source.rotation)
                return PixelSize(width: size.width, height: size.height)
            }()
        }
        monitor.updateContext {
            $0.sourceSize = source.size
            $0.aiInputSize = aiInput
            $0.outputSize = output
            $0.nominalFPS = source.nominalFrameRate
        }
    }

    // MARK: - Dynamic quality

    private func evaluateDynamicQuality(config: ProcessingConfiguration, models: Models, playing: Bool, rate: Double,
                                        interval: CFTimeInterval, now: CFTimeInterval) {
        guard now - lastEvaluation >= 0.25 else { return }
        lastEvaluation = now
        // Throughput measured across a pause means nothing: restart the grace period on resume.
        if !playing { aiActiveSince = nil }
        let model = config.mode == .aiHighQuality ? models.highQuality : models.realtime
        let snapshot = monitor.aiStatistics
        let nominal = lastSource?.nominalFrameRate ?? 0
        let required = (nominal > 0 ? nominal : 30) * max(abs(rate), playing ? 0.01 : 0)
        let inputs = DynamicQualityController.Inputs(
            mode: config.mode, dynamicQuality: config.dynamicQuality, playing: playing,
            modelReady: model != nil, modelProblem: models.problem,
            sourceHeight: lastSource?.size.height ?? 0, requiredFPS: required,
            aiThroughput: snapshot.throughput, aiLatencyMedian: snapshot.latencyMedian,
            aiActiveSince: aiActiveSince, fastGPUms: monitor.renderGPUMedian, frameBudgetMs: interval * 1000, now: now
        )
        let (old, new) = lock.withLock { () -> (DynamicQualityController.Decision, DynamicQualityController.Decision) in
            let old = dynamic.decision
            return (old, dynamic.evaluate(inputs))
        }
        if new.path != .ai { aiActiveSince = nil }
        if old.path != new.path { lock.withLock { generation += 1 } }

        let label: String = switch new.path {
        case .original: "Original"
        case .fast: "Fast · Metal"
        case .ai: model.map { "AI · \($0.info.displayName)" } ?? "AI"
        }
        monitor.updateContext {
            $0.pathLabel = label
            $0.pathReason = new.reason
            $0.modelName = model?.info.displayName
            $0.computeUnits = (model as? CoreMLSuperResolution)?.computeUnits.title
            $0.playbackRate = rate
        }
    }
}
