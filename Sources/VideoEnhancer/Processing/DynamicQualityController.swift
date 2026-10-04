import Foundation
import QuartzCore

/// Chooses the processing path that keeps playback smooth: smoothness beats maximum quality.
///
/// Ladder for the "AI" mode: AI (at full source resolution) → Fast (Metal) → Original.
/// It never feeds the model a smaller-than-source input to gain speed: measured on real photos,
/// downscale + AI is worse than Lanczos (Docs/02_MODEL_EVALUATION.md, "reduced input").
struct DynamicQualityController {
    enum Path: Equatable, Sendable {
        case original
        case fast
        case ai
    }

    struct Decision: Equatable, Sendable {
        var path: Path
        var reason: String?
    }

    struct Inputs {
        var mode: QualityMode
        var dynamicQuality: Bool
        var playing: Bool
        var modelReady: Bool
        var modelProblem: String?
        var sourceHeight: Int
        /// Frames per second the display needs: nominal fps × |rate|.
        var requiredFPS: Double
        var aiThroughput: Double
        var aiLatencyMedian: Double?   // ms
        var aiActiveSince: CFTimeInterval?
        var fastGPUms: Double?
        var frameBudgetMs: Double
        var now: CFTimeInterval
    }

    private(set) var decision = Decision(path: .fast, reason: nil)
    private var behindSince: CFTimeInterval?
    private var fallbackUntil: CFTimeInterval = 0
    private var fallbackCount = 0
    private var fallbackRequiredFPS: Double = 0

    /// Real-time AI is applied up to 720p sources: above that no model on the M4 Neural Engine reaches
    /// real time (SPAN ×2 at 720p: ~12 fps, Docs/03) and the source already carries most of the detail.
    static let maxRealtimeAISourceHeight = 720
    /// AI High Quality accepts up to 1080p sources (largest enumerated model input).
    static let maxHighQualitySourceHeight = 1080

    mutating func reset() {
        behindSince = nil
        fallbackUntil = 0
        fallbackCount = 0
        fallbackRequiredFPS = 0
        decision = Decision(path: .fast, reason: nil)
    }

    mutating func evaluate(_ input: Inputs) -> Decision {
        decision = decide(input)
        return decision
    }

    private mutating func decide(_ i: Inputs) -> Decision {
        switch i.mode {
        case .original:
            return Decision(path: .original)
        case .fast:
            return fastOrOriginal(i)
        case .aiHighQuality:
            if let problem = i.modelProblem { return Decision(path: .fast, reason: problem) }
            if !i.modelReady { return Decision(path: .fast, reason: "Loading AI model…") }
            if i.sourceHeight > Self.maxHighQualitySourceHeight {
                return Decision(path: .fast, reason: "Source is \(i.sourceHeight)p — AI High Quality accepts up to \(Self.maxHighQualitySourceHeight)p")
            }
            return Decision(path: .ai, reason: i.playing && i.aiThroughput > 0 && i.aiThroughput < i.requiredFPS * 0.95
                ? "Not real time: \(Int(i.aiThroughput.rounded())) of \(Int(i.requiredFPS.rounded())) fps" : nil)
        case .ai:
            break
        }

        if let problem = i.modelProblem { return Decision(path: .fast, reason: problem) }
        if !i.modelReady { return Decision(path: .fast, reason: "Loading AI model…") }
        if i.sourceHeight > Self.maxRealtimeAISourceHeight {
            return Decision(path: .fast, reason: "Source is \(i.sourceHeight)p — real-time AI is applied up to \(Self.maxRealtimeAISourceHeight)p (use AI High Quality)")
        }
        // A paused frame has no deadline: always show the AI result.
        if !i.playing || !i.dynamicQuality { behindSince = nil; return Decision(path: .ai) }

        // Previously fell back: retry when the cooldown has passed or the demand dropped (lower rate).
        if decision.path != .ai, fallbackUntil > i.now, i.requiredFPS >= fallbackRequiredFPS * 0.9 {
            return decision
        }
        if decision.path != .ai, let latency = i.aiLatencyMedian, fallbackCount > 0 {
            // Estimate from the last measurement: two frames in flight overlap pre/post-processing with inference.
            let estimate = 1000 / latency * 1.3
            if estimate < i.requiredFPS * 0.95, i.requiredFPS >= fallbackRequiredFPS * 0.9 {
                fallbackUntil = i.now + 10
                return decision
            }
        }

        // Running AI: give it a moment to fill the pipeline, then compare throughput with demand.
        if let since = i.aiActiveSince, i.now - since > 1.5, i.aiThroughput < i.requiredFPS * 0.92 {
            if behindSince == nil { behindSince = i.now }
            if i.now - (behindSince ?? i.now) > 1.0 {
                behindSince = nil
                fallbackCount += 1
                fallbackRequiredFPS = i.requiredFPS
                fallbackUntil = i.now + min(60, 10 * Double(fallbackCount))
                let reason = "AI \(String(format: "%.1f", i.aiThroughput)) fps < \(String(format: "%.0f", i.requiredFPS)) fps needed → Fast"
                return Decision(path: .fast, reason: reason)
            }
        } else {
            behindSince = nil
        }
        return Decision(path: .ai)
    }

    private func fastOrOriginal(_ i: Inputs) -> Decision {
        if i.dynamicQuality, let gpu = i.fastGPUms, gpu > i.frameBudgetMs * 0.8 {
            return Decision(path: .original, reason: "Fast needs \(String(format: "%.1f", gpu)) ms > frame budget")
        }
        return Decision(path: .fast)
    }
}
