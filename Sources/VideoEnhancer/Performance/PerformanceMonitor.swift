import Foundation
import QuartzCore

/// Collects pipeline measurements from the render thread, the AI completion handlers and Metal
/// command buffer callbacks. Thread-safe; `snapshot()` is cheap enough to call a few times a second.
final class PerformanceMonitor: @unchecked Sendable {
    private let lock = NSLock()

    private var pulled = FPSCounter()
    private var presented = FPSCounter()
    private var aiCompleted = FPSCounter(window: 2)
    private var latency = SampleWindow()
    private var avOffset = SampleWindow()
    private var inference = SampleWindow()
    private var preprocess = SampleWindow()
    private var postprocess = SampleWindow()
    private var aiTotal = SampleWindow()
    private var renderGPU = SampleWindow()
    private var gpuBusy: [(end: CFTimeInterval, duration: CFTimeInterval)] = []
    private var drops: [DropReason: Int] = [:]
    private var cpuSampler = CPUUsageSampler()
    private var lastCPU: Double?
    private var lastCPUSampleTime: CFTimeInterval = 0

    // Context set by the engine.
    private var context = PerformanceSnapshot()

    // MARK: - Recording

    func recordPulled(at time: CFTimeInterval) {
        lock.withLock { pulled.record(time) }
    }

    func recordPresented(pulledAt: CFTimeInterval, dueAt: CFTimeInterval, presentedAt: CFTimeInterval) {
        lock.withLock {
            presented.record(presentedAt)
            latency.add((presentedAt - pulledAt) * 1000)
            avOffset.add((presentedAt - dueAt) * 1000)
        }
    }

    var renderGPUMedian: Double? { lock.withLock { renderGPU.median } }
    var inferenceMean: Double? { lock.withLock { inference.mean } }

    /// Share of `seconds` (ending now) that this app's command buffers ran on the GPU.
    func gpuBusyPercent(over seconds: Double) -> Double {
        let now = CACurrentMediaTime()
        return lock.withLock {
            min(100, gpuBusy.filter { now - $0.end <= seconds }.reduce(0) { $0 + $1.duration } / seconds * 100)
        }
    }

    func recordAI(_ timings: SRTimings) {
        lock.withLock {
            aiCompleted.record()
            inference.add(timings.inference * 1000)
            preprocess.add(timings.preprocessGPU * 1000)
            postprocess.add(timings.postprocessGPU * 1000)
            aiTotal.add(timings.total * 1000)
        }
    }

    func recordGPU(start: CFTimeInterval, end: CFTimeInterval, isRenderPass: Bool) {
        guard end > start, start > 0 else { return }
        lock.withLock {
            gpuBusy.append((end, end - start))
            if gpuBusy.count > 20_000 { gpuBusy.removeFirst(gpuBusy.count - 20_000) }
            if isRenderPass { renderGPU.add((end - start) * 1000) }
        }
    }

    func recordDrop(_ reason: DropReason, count: Int = 1) {
        guard count > 0 else { return }
        lock.withLock { drops[reason, default: 0] += count }
    }

    /// Engine-provided description of what is running (path, model, sizes, rates).
    func updateContext(_ update: (inout PerformanceSnapshot) -> Void) {
        lock.withLock { update(&context) }
    }

    func reset() {
        lock.withLock {
            pulled.reset(); presented.reset(); aiCompleted.reset()
            latency.reset(); avOffset.reset(); inference.reset(); preprocess.reset(); postprocess.reset(); aiTotal.reset(); renderGPU.reset()
            gpuBusy.removeAll(); drops.removeAll()
        }
    }

    // MARK: - Reading

    /// Median AI latency (submit → ready) and throughput, used by Dynamic Quality and the lookahead.
    var aiStatistics: (latencyMedian: Double?, latencyP95: Double?, inferenceMedian: Double?, throughput: Double) {
        lock.withLock { (aiTotal.median, aiTotal.p95, inference.median, aiCompleted.rate()) }
    }

    func snapshot() -> PerformanceSnapshot {
        let now = CACurrentMediaTime()
        return lock.withLock {
            var s = context
            s.videoFPS = pulled.rate(now: now)
            s.displayFPS = presented.rate(now: now)
            s.processingFPS = aiCompleted.rate(now: now)
            s.latencyMs = latency.median
            s.avOffsetMs = avOffset.median
            s.inferenceMs = inference.median
            s.inferenceP95Ms = inference.p95
            s.preprocessGPUMs = preprocess.median
            s.postprocessGPUMs = postprocess.median
            s.renderGPUMs = renderGPU.median
            s.frameTimeMs = s.pathLabel.hasPrefix("AI") ? aiTotal.median : renderGPU.median
            s.dropped = drops

            let recent = gpuBusy.filter { now - $0.end <= 1.0 }
            s.gpuBusyPercent = recent.isEmpty ? 0 : min(100, recent.reduce(0) { $0 + $1.duration } * 100)

            if now - lastCPUSampleTime > 0.5 {
                lastCPU = cpuSampler.sample() ?? lastCPU
                lastCPUSampleTime = now
            }
            s.cpuPercent = lastCPU
            s.memoryBytes = SystemMetrics.memoryFootprintBytes()
            s.thermalState = SystemMetrics.thermalState
            return s
        }
    }
}
