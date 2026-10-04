import AppKit
import SwiftUI

@MainActor
@Observable
final class BenchmarkController {
    /// nil = Metal Fast path.
    var modelID: String? = "span-nomosuni-x2"
    var inputSize = PixelSize(width: 1280, height: 720)
    var duration: Double = 10
    var arrivalFPS: Double = 60
    var computeUnits: ComputeUnitsOption = .cpuAndNeuralEngine
    var compareAllUnits = false
    var useCurrentVideo = true

    private(set) var isRunning = false
    private(set) var progress = 0.0
    private(set) var status = ""
    private(set) var results: [BenchmarkResult] = []

    static let inputChoices = [PixelSize(width: 640, height: 360), PixelSize(width: 960, height: 540),
                               PixelSize(width: 1280, height: 720), PixelSize(width: 1920, height: 1080)]

    func run(player: VideoPlayer) async {
        guard !isRunning, case .success(let context) = MetalContext.shared else { return }
        isRunning = true
        defer { isRunning = false; progress = 0 }
        let wasPlaying = player.playback.isPlaying
        if wasPlaying { player.playback.togglePlayPause() }

        let runner = BenchmarkRunner(context: context)
        let manager = player.enhancement.modelManager
        let units: [ComputeUnitsOption] = modelID == nil ? [.cpuAndNeuralEngine]
            : compareAllUnits ? [.cpuAndNeuralEngine, .all, .cpuAndGPU] : [computeUnits]
        for unit in units {
            var config = BenchmarkConfiguration()
            config.modelID = modelID
            config.computeUnits = unit
            config.inputSize = inputSize
            config.metalOutputSize = inputSize.scaled(by: 2)
            config.duration = duration
            config.arrivalFPS = arrivalFPS
            config.sourceFile = useCurrentVideo ? player.playback.playlist.current : nil
            config.postprocessing = AIPostSettings(consistency: player.enhancement.configuration.consistencyCorrection,
                                                   stabilization: player.enhancement.configuration.temporalStabilization,
                                                   sharpen: player.enhancement.configuration.aiSharpen)

            var model: CoreMLSuperResolution?
            var loadSeconds: Double?
            if let modelID {
                status = "Loading \(modelID) (\(unit.shortTitle))…"
                let start = CFAbsoluteTimeGetCurrent()
                do {
                    model = try await manager.load(id: modelID, units: unit)
                    loadSeconds = CFAbsoluteTimeGetCurrent() - start
                } catch {
                    status = "Load failed: \(error.localizedDescription)"
                    continue
                }
            }
            status = "Running \(modelID ?? "Metal Fast") · \(unit.shortTitle) · \(inputSize)…"
            let result = await runner.run(config, model: model, loadSeconds: loadSeconds) { value in
                Task { @MainActor [weak self] in self?.progress = value }
            }
            results.insert(result, at: 0)
        }
        status = "Done"
        if wasPlaying { player.playback.togglePlayPause() }
    }

    var resultsText: String {
        results.map { r in
            "\(r.modelName) | \(r.configuration.modelID == nil ? "GPU" : r.configuration.computeUnits.shortTitle) | "
                + "\(r.configuration.inputSize) → \(r.outputSize) | avg \(String(format: "%.1f", r.averageFPS)) fps | "
                + "inference \(r.inferenceMedianMs.map { String(format: "%.1f", $0) } ?? "—") ms | "
                + "dropped \(r.framesDropped)/\(r.framesOffered) | peak \(r.peakMemoryBytes / 1_048_576) MB"
        }.joined(separator: "\n")
    }
}

struct BenchmarkView: View {
    let player: VideoPlayer
    @Bindable var benchmark: BenchmarkController
    @Environment(\.dismiss) private var dismiss

    private var manager: ModelManager { player.enhancement.modelManager }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Benchmark").font(.title2.weight(.semibold))
            Text("Runs the playback pipeline (color conversion → model / Metal → post-processing) on frames arriving at a fixed rate. Frames that arrive while the pipeline is full are dropped, as during playback.")
                .font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Form {
                Picker("Model", selection: $benchmark.modelID) {
                    ForEach(manager.entries) { Text($0.info.label).tag(Optional($0.id)) }
                    Text("Metal Fast (classical, not AI)").tag(String?.none)
                }
                Picker("Input", selection: $benchmark.inputSize) {
                    ForEach(availableInputs, id: \.self) { Text($0.description).tag($0) }
                }
                LabeledContent("Output", value: outputSize.description)
                Picker("Duration", selection: $benchmark.duration) {
                    ForEach([5.0, 10, 30], id: \.self) { Text("\(Int($0)) seconds").tag($0) }
                }
                Picker("Frames arrive at", selection: $benchmark.arrivalFPS) {
                    ForEach([24.0, 30, 60], id: \.self) { Text("\(Int($0)) fps").tag($0) }
                }
                if benchmark.modelID != nil {
                    Picker("Compute", selection: $benchmark.computeUnits) {
                        ForEach(ComputeUnitsOption.allCases) { Text($0.title).tag($0) }
                    }
                    .disabled(benchmark.compareAllUnits)
                    Toggle("Compare CPU+ANE, ALL and CPU+GPU", isOn: $benchmark.compareAllUnits)
                }
                Picker("Source", selection: $benchmark.useCurrentVideo) {
                    Text("Current video").tag(true).disabled(player.playback.playlist.current == nil)
                    Text("Synthetic pattern").tag(false)
                }
            }
            .formStyle(.grouped)
            .frame(height: 330)

            HStack {
                Button(benchmark.isRunning ? "Running…" : "Run Benchmark") {
                    Task { await benchmark.run(player: player) }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(benchmark.isRunning)
                if benchmark.isRunning { ProgressView(value: benchmark.progress).frame(width: 160) }
                Text(benchmark.status).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                Spacer()
                Button("Copy Results") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(benchmark.resultsText, forType: .string)
                }
                .disabled(benchmark.results.isEmpty)
                Button("Close") { dismiss() }
            }

            Table(benchmark.results.indices.map { IndexedResult(index: $0, result: benchmark.results[$0]) }) {
                TableColumn("Model") { Text($0.result.modelName).help($0.result.error ?? "") }.width(min: 150)
                TableColumn("Compute") { Text($0.result.configuration.modelID == nil ? "GPU" : $0.result.configuration.computeUnits.shortTitle) }.width(60)
                TableColumn("Input → Output") { Text($0.result.configuration.inputSize.description + " → " + $0.result.outputSize.description) }.width(min: 130)
                TableColumn("Avg FPS") { Text(String(format: "%.1f", $0.result.averageFPS)).monospacedDigit() }.width(55)
                TableColumn("Inference") { item in
                    Text("\(ms(item.result.inferenceMedianMs)) / \(ms(item.result.inferenceP95Ms))").monospacedDigit()
                        .help("Median / p95. For Metal Fast: GPU time per frame.")
                }.width(min: 90)
                TableColumn("Dropped") { Text("\($0.result.framesDropped)/\($0.result.framesOffered)").monospacedDigit() }.width(70)
                TableColumn("Peak mem") { Text("\($0.result.peakMemoryBytes / 1_048_576) MB").monospacedDigit() }.width(70)
                TableColumn("CPU") { Text($0.result.cpuPercent.map { "\(Int($0))%" } ?? "—").monospacedDigit() }.width(45)
                TableColumn("GPU") { Text($0.result.gpuBusyPercent.map { "\(Int($0))%" } ?? "—").monospacedDigit() }.width(45)
                TableColumn("Real time") { item in
                    Text(item.result.error != nil ? "error" : item.result.isRealTime ? "yes" : "no")
                        .foregroundStyle(item.result.isRealTime ? .green : .orange)
                }.width(60)
            }
            .frame(minHeight: 160)

            Text("CPU: this process's CPU time ÷ wall time (100% = one core). GPU: share of time this app's Metal command buffers ran (Core ML's own GPU work is not included). Neural Engine utilization is not available through public macOS APIs; inference time is the measurable proxy.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(20)
        .frame(width: 1020, height: 760)
    }

    private var availableInputs: [PixelSize] {
        guard let id = benchmark.modelID, let entry = manager.entry(id: id) else { return BenchmarkController.inputChoices }
        return BenchmarkController.inputChoices.filter(entry.info.inputSizes.contains)
    }

    private var outputSize: PixelSize {
        let scale = benchmark.modelID.flatMap { manager.entry(id: $0)?.info.scale } ?? 2
        return benchmark.inputSize.scaled(by: scale)
    }

    private func ms(_ v: Double?) -> String { v.map { String(format: "%.1f", $0) } ?? "—" }
}

private struct IndexedResult: Identifiable {
    let index: Int
    let result: BenchmarkResult
    var id: Int { index }
}
