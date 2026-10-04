import Foundation

/// Why a frame never reached the screen.
enum DropReason: String, CaseIterable, Sendable {
    /// A newer frame arrived while the AI was busy; the older pending one was discarded (backpressure).
    case superseded = "Superseded while AI busy"
    /// Finished processing, but a newer frame was already due — shown frames stay in order.
    case late = "Finished too late"
    /// Never pulled from the decoder because display/processing was slower than the source.
    case notPulled = "Not pulled (decoder skipped ahead)"
    /// Processing failed.
    case failed = "Processing error"
}

/// Timings of one AI-processed frame.
struct SRTimings: Sendable {
    /// GPU time of the input tensor preparation (MTLCommandBuffer GPU start → end).
    var preprocessGPU: Double
    /// Wall time of the Core ML prediction call.
    var inference: Double
    /// GPU time of output unpacking, consistency correction, stabilization, sharpening.
    var postprocessGPU: Double
    /// Submit → result ready, wall time.
    var total: Double
    var inputSize: PixelSize
    var outputSize: PixelSize
}

/// One coherent set of numbers for the UI and the benchmark. Every field is measured; fields that
/// cannot be measured with public APIs are absent rather than estimated.
struct PerformanceSnapshot: Sendable {
    // Video
    var sourceSize: PixelSize?
    var outputSize: PixelSize?
    var nominalFPS: Double = 0
    var playbackRate: Double = 0
    /// New decoded frames pulled per second.
    var videoFPS: Double = 0
    /// Distinct frames presented per second.
    var displayFPS: Double = 0
    /// AI frames completed per second.
    var processingFPS: Double = 0
    var vsyncRate: Double = 0

    // Timing (milliseconds)
    var frameTimeMs: Double?          // AI: submit → ready (median); Fast/Original: GPU time per frame
    var inferenceMs: Double?          // median Core ML prediction
    var inferenceP95Ms: Double?
    var preprocessGPUMs: Double?
    var postprocessGPUMs: Double?
    var renderGPUMs: Double?          // per-vsync command buffer (convert + Fast + present)
    var latencyMs: Double?            // pulled → actually on screen (median)
    var avOffsetMs: Double?           // on screen − due time (median); ~0 means in sync with audio
    var lookaheadMs: Double = 0

    // Drops
    var dropped: [DropReason: Int] = [:]
    var droppedTotal: Int { dropped.values.reduce(0, +) }

    // Pipeline
    var pathLabel = "—"
    var pathReason: String?
    var modelName: String?
    var computeUnits: String?
    var aiInputSize: PixelSize?

    // System
    var cpuPercent: Double?           // 100 = one core
    var memoryBytes: UInt64?
    var gpuBusyPercent: Double?       // share of wall time our Metal command buffers ran on the GPU
    var thermalState = SystemMetrics.thermalState
}
