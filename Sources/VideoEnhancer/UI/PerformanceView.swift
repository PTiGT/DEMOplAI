import SwiftUI

/// Technical panel. Every number is measured by the app (or read from AVFoundation / Core Video /
/// Core ML); metrics without a public API are labelled as unavailable instead of being estimated.
struct PerformanceView: View {
    let player: VideoPlayer

    @State private var frameStats = FrameProvider.Statistics()

    private var s: PerformanceSnapshot { player.enhancement.snapshot }

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 2) {
            header("VIDEO")
            row("Resolution", s.sourceSize?.description ?? "—")
            row("Output", s.outputSize.map { "\($0)" + (s.aiInputSize.map { "  (AI input \($0))" } ?? "") } ?? "—")
            row("Video FPS", "\(fps(s.videoFPS)) pulled · nominal \(fps(s.nominalFPS))" + (abs(s.playbackRate - 1) > 0.01 && s.playbackRate > 0 ? " · \(fps(s.playbackRate))×" : ""))
            row("Display FPS", "\(fps(s.displayFPS)) · vsync \(Int(frameStats.vsyncRate.rounded())) Hz")
            row("Processing FPS", s.pathLabel.hasPrefix("AI") ? fps(s.processingFPS) : "— (no AI running)")

            header("TIMING")
            row("Frame Time", ms(s.frameTimeMs) + (s.pathLabel.hasPrefix("AI") ? " (AI submit → ready)" : " (GPU, per vsync)"))
            if s.pathLabel.hasPrefix("AI") {
                row("Inference", "\(ms(s.inferenceMs)) median · \(ms(s.inferenceP95Ms)) p95")
                row("Pre / Post GPU", "\(ms(s.preprocessGPUMs)) / \(ms(s.postprocessGPUMs))")
                row("Lookahead", ms(s.lookaheadMs))
            }
            row("Latency", ms(s.latencyMs) + " (pulled → on screen)")
            row("A/V offset", ms(s.avOffsetMs) + " (on screen − due)")
            row("Dropped", "\(s.droppedTotal + Int(frameStats.framesNotPulled))" + dropDetail)

            header("PIPELINE")
            row("Pipeline", pipelineText)
            if let reason = s.pathReason { row("", reason) }
            row("AI Model", player.enhancement.modelStatus)
            row("Compute", s.computeUnits ?? "—")
            if let plan = currentPlan { row("Compute plan", plan.label) }

            header("SYSTEM")
            row("Memory", s.memoryBytes.map { ByteCountFormatter.string(fromByteCount: Int64($0), countStyle: .memory) } ?? "—")
            row("CPU", s.cpuPercent.map { "\(Int($0.rounded()))% of one core (\(SystemMetrics.cpuCoreCount) cores)" } ?? "—")
            row("GPU", s.gpuBusyPercent.map { "\(Int($0.rounded()))% busy with this app's Metal work" } ?? "—")
            row("Neural Engine", "utilization not exposed by macOS (see compute plan)")
            row("Thermal", s.thermalState)
        }
        .font(.system(size: 11, design: .monospaced))
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
        .environment(\.colorScheme, .dark)
        .task {
            while !Task.isCancelled {
                frameStats = player.playback.frameProvider.statistics()
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
    }

    private var pipelineText: String {
        let decode = frameStats.lastFrame.map { "VideoToolbox (\(PixelFormat.fourCC($0.pixelFormat)))" } ?? "VideoToolbox"
        switch s.pathLabel {
        case let label where label.hasPrefix("AI"): return "\(decode) → Metal → Core ML → Metal"
        case let label where label.hasPrefix("Fast"): return "\(decode) → Metal (enhance) → Metal"
        default: return "\(decode) → Metal"
        }
    }

    private var currentPlan: ModelManager.ComputePlanSummary? {
        let id = player.enhancement.configuration.mode == .aiHighQuality
            ? player.enhancement.activeHQModel?.id : player.enhancement.activeRealtimeModel?.id
        return id.flatMap { player.enhancement.modelManager.entry(id: $0)?.computePlan }
    }

    private var dropDetail: String {
        var parts = s.dropped.filter { $0.value > 0 }.map { "\($0.key.rawValue.lowercased()) \($0.value)" }
        if frameStats.framesNotPulled > 0 { parts.append("not pulled \(frameStats.framesNotPulled)") }
        return parts.isEmpty ? "" : " (" + parts.sorted().joined(separator: ", ") + ")"
    }

    private func header(_ title: String) -> some View {
        Text(title).font(.system(size: 9, weight: .semibold)).foregroundStyle(.secondary).padding(.top, 4)
    }

    private func row(_ title: String, _ value: String) -> some View {
        GridRow {
            Text(title).foregroundStyle(.secondary)
            Text(value).lineLimit(1)
        }
    }

    private func fps(_ v: Double) -> String { v.formatted(.number.precision(.fractionLength(0...1))) }
    private func ms(_ v: Double?) -> String { v.map { "\($0.formatted(.number.precision(.fractionLength(1)))) ms" } ?? "—" }
}
