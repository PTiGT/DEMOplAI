import SwiftUI

/// "Quality Boost" popover: the four modes plus Dynamic Quality.
struct QualityMenuView: View {
    @Bindable var enhancement: EnhancementController

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Quality Boost").font(.headline)
            Picker("Quality Boost", selection: $enhancement.configuration.mode) {
                ForEach(QualityMode.allCases) { mode in
                    VStack(alignment: .leading, spacing: 1) {
                        Text(mode.title)
                        Text(mode.detail).font(.caption).foregroundStyle(.secondary)
                    }
                    .tag(mode)
                }
            }
            .pickerStyle(.radioGroup)
            .labelsHidden()
            Divider()
            Toggle("Dynamic Quality", isOn: $enhancement.configuration.dynamicQuality)
            Text("Drops to a lighter mode when the selected one cannot keep up with the video.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .frame(width: 320)
    }
}

/// Advanced settings and live numbers (the "⚙" popover).
struct SettingsView: View {
    let player: VideoPlayer
    @Bindable var enhancement: EnhancementController

    init(player: VideoPlayer) {
        self.player = player
        enhancement = player.enhancement
    }

    private var manager: ModelManager { enhancement.modelManager }
    private var s: PerformanceSnapshot { enhancement.snapshot }

    var body: some View {
        Form {
            Section("AI") {
                Picker("AI Model", selection: $enhancement.configuration.aiModelFamily) {
                    ForEach(manager.families, id: \.family) { Text($0.name).tag($0.family) }
                }
                Picker("High Quality Model", selection: $enhancement.configuration.hqModelFamily) {
                    ForEach(manager.families, id: \.family) { Text($0.name).tag($0.family) }
                }
                Picker("Compute", selection: $enhancement.configuration.computeUnits) {
                    ForEach(ComputeUnitsOption.allCases) { Text($0.title).tag($0) }
                }
                Toggle("Low-frequency consistency", isOn: $enhancement.configuration.consistencyCorrection)
                    .help("Keeps brightness, color and coarse structure of the AI output equal to the source.")
                Toggle("Temporal stabilization", isOn: $enhancement.configuration.temporalStabilization)
                    .help("Blends with the previous AI frame where the source did not change: less flicker.")
                Slider(value: $enhancement.configuration.aiSharpen, in: 0...1) { Text("AI sharpening") }
            }

            Section("Fast (Metal)") {
                Slider(value: $enhancement.configuration.enhancement.sharpness, in: 0...1) { Text("Sharpness") }
                Slider(value: $enhancement.configuration.enhancement.denoise, in: 0...1) { Text("Artifact reduction") }
            }

            Section("Live") {
                LabeledContent("AI Model", value: enhancement.modelStatus)
                LabeledContent("Scale", value: activeInfo.map { "\($0.scale)×" } ?? "—")
                LabeledContent("Compute", value: s.computeUnits ?? "—")
                LabeledContent("Path", value: s.pathLabel + (s.pathReason.map { " — \($0)" } ?? ""))
                LabeledContent("Processing", value: s.pathLabel.hasPrefix("AI") ? "\(fps(s.processingFPS)) FPS" : "—")
                LabeledContent("Video", value: "\(fps(s.videoFPS)) FPS")
                LabeledContent("Dropped", value: "\(s.droppedTotal)")
            }

            Section("Installed models") {
                if manager.entries.isEmpty {
                    Text(manager.isDiscovering ? "Searching…" : "No models found in Models/ or Application Support.")
                        .foregroundStyle(.secondary)
                }
                ForEach(manager.entries) { entry in
                    ModelRow(entry: entry)
                }
                ForEach(manager.discoveryErrors, id: \.self) { Text($0).foregroundStyle(.red).font(.caption) }
                HStack {
                    Button("Reload Models") { enhancement.reloadModels() }
                    Button("Unload") { enhancement.unloadModels() }
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 460, height: 620)
    }

    private var activeInfo: SuperResolutionModelInfo? {
        enhancement.configuration.mode == .aiHighQuality ? enhancement.activeHQModel : enhancement.activeRealtimeModel
    }

    private func fps(_ v: Double) -> String { v.formatted(.number.precision(.fractionLength(0...1))) }
}

private struct ModelRow: View {
    let entry: ModelManager.Entry

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(entry.info.label).font(.body.weight(.medium))
                Spacer()
                Text(entry.state.label).font(.caption).foregroundStyle(.secondary)
            }
            Group {
                Text("\(entry.info.architecture) · \(entry.info.parameters.map { "\($0.formatted()) params" } ?? "") · v\(entry.info.version) · \(ByteCountFormatter.string(fromByteCount: Int64(entry.sizeBytes), countStyle: .file))")
                Text("\(entry.info.license) · \(entry.info.source)")
                Text("Inputs: " + entry.info.inputSizes.map(\.description).joined(separator: ", "))
                if let load = entry.loadSeconds {
                    Text("Load \(String(format: "%.2f", load)) s" + entry.warmUpSeconds
                        .sorted { $0.key.pixels < $1.key.pixels }
                        .map { " · first run \($0.key): \(String(format: "%.2f", $0.value)) s" }.joined())
                }
                if let plan = entry.computePlan { Text("Plan: " + plan.label) }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }
}
