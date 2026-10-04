import CoreML
import CryptoKit
import Foundation
import Observation

/// Finds, compiles, validates, loads, warms up and unloads super-resolution models.
///
/// Models are loaded once per (model, compute units) and shared; nothing is loaded per frame.
/// Search order: app bundle `Resources/Models`, `~/Library/Application Support/VideoEnhancer/Models`,
/// and the repository's `Models/` folder (development runs).
@MainActor
@Observable
final class ModelManager {
    enum State: Equatable, Sendable {
        case available           // found and compiled, not loaded
        case loading
        case loaded(ComputeUnitsOption)
        case failed(String)

        var label: String {
            switch self {
            case .available: "Available"
            case .loading: "Loading…"
            case .loaded(let units): "Loaded (\(units.shortTitle))"
            case .failed(let reason): "Failed: \(reason)"
            }
        }
    }

    struct ComputePlanSummary: Sendable, Equatable {
        var operations = 0
        var neuralEngine = 0
        var gpu = 0
        var cpu = 0
        /// Share of the estimated cost assigned to the Neural Engine (Core ML's own estimate).
        var neuralEngineCostShare: Double?

        var label: String {
            var text = "ANE \(neuralEngine) · GPU \(gpu) · CPU \(cpu) of \(operations) ops"
            if let share = neuralEngineCostShare { text += " (\(Int((share * 100).rounded()))% of est. cost on ANE)" }
            return text
        }
    }

    struct Entry: Identifiable, Equatable {
        var id: String { info.id }
        var info: SuperResolutionModelInfo
        var compiledURL: URL
        var sourceURL: URL
        var sizeBytes: Int
        var compileSeconds: Double?
        var state: State = .available
        var loadSeconds: Double?
        /// First prediction per input size: includes on-device specialization (e.g. ANE compilation).
        var warmUpSeconds: [PixelSize: Double] = [:]
        var computePlan: ComputePlanSummary?
    }

    private(set) var entries: [Entry] = []
    private(set) var isDiscovering = false
    private(set) var discoveryErrors: [String] = []

    @ObservationIgnored private var loaded: [String: CoreMLSuperResolution] = [:]
    @ObservationIgnored private var loadTasks: [String: Task<CoreMLSuperResolution, Error>] = [:]

    // MARK: - Discovery

    nonisolated static var searchDirectories: [URL] {
        var dirs: [URL] = []
        if let resources = Bundle.main.resourceURL { dirs.append(resources.appendingPathComponent("Models")) }
        if let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            dirs.append(support.appendingPathComponent("VideoEnhancer/Models"))
        }
        // Development: <repo>/Models relative to this source file.
        dirs.append(URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Models"))
        return dirs
    }

    func discover() async {
        guard !isDiscovering else { return }
        isDiscovering = true
        defer { isDiscovering = false }

        let found = await Task.detached(priority: .utility) { () -> ([Entry], [String]) in
            var entries: [String: Entry] = [:]
            var errors: [String] = []
            for directory in Self.searchDirectories {
                let items = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
                for url in items where ["mlpackage", "mlmodelc"].contains(url.pathExtension) {
                    do {
                        let entry = try await Self.prepare(url)
                        if entries[entry.id] == nil { entries[entry.id] = entry }  // earlier directories win
                    } catch {
                        errors.append("\(url.lastPathComponent): \(error.localizedDescription)")
                    }
                }
            }
            return (entries.values.sorted { ($0.info.family, $0.info.scale) < ($1.info.family, $1.info.scale) }, errors)
        }.value

        // Keep runtime state of models that are already loaded.
        entries = found.0.map { entry in
            var entry = entry
            if let old = entries.first(where: { $0.id == entry.id }) {
                entry.state = old.state
                entry.loadSeconds = old.loadSeconds
                entry.warmUpSeconds = old.warmUpSeconds
                entry.computePlan = old.computePlan
            }
            return entry
        }
        discoveryErrors = found.1
    }

    /// Compiles an .mlpackage (cached by content) and reads + validates its metadata.
    nonisolated private static func prepare(_ url: URL) async throws -> Entry {
        var compiled = url
        var compileSeconds: Double?
        if url.pathExtension == "mlpackage" {
            let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("VideoEnhancer/CompiledModels", isDirectory: true)
            try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
            let target = cacheDir.appendingPathComponent("\(url.deletingPathExtension().lastPathComponent)-\(contentHash(url)).mlmodelc")
            if !FileManager.default.fileExists(atPath: target.path) {
                let start = CFAbsoluteTimeGetCurrent()
                let temporary = try await MLModel.compileModel(at: url)
                try? FileManager.default.removeItem(at: target)
                try FileManager.default.moveItem(at: temporary, to: target)
                compileSeconds = CFAbsoluteTimeGetCurrent() - start
            }
            compiled = target
        }
        let info = try readInfo(compiledURL: compiled)
        return Entry(info: info, compiledURL: compiled, sourceURL: url, sizeBytes: directorySize(url), compileSeconds: compileSeconds)
    }

    /// Reads the model metadata and checks the I/O contract the pipeline relies on. Uses Core ML's own
    /// model description (models compiled at runtime have no metadata.json); the CPU-only load used
    /// here does not compile anything for the Neural Engine.
    nonisolated static func readInfo(compiledURL: URL) throws -> SuperResolutionModelInfo {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        let description = try MLModel(contentsOf: compiledURL, configuration: configuration).modelDescription
        let user = description.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        guard let id = user["ve.id"], let scale = user["ve.scale"].flatMap(Int.init), scale > 1 else {
            throw SRError.incompatible("missing ve.* metadata (not converted by Tools/ml/convert_to_coreml.py)")
        }
        guard let input = description.inputDescriptionsByName["input"]?.multiArrayConstraint, input.dataType == .float16,
              let output = description.outputDescriptionsByName["output"]?.multiArrayConstraint, output.dataType == .float16
        else {
            throw SRError.incompatible("expected Float16 multi-array tensors named input/output")
        }
        let sizes = (user["ve.inputShapes"] ?? "").split(separator: ",").compactMap { token -> PixelSize? in
            let parts = token.split(separator: "x").compactMap { Int($0) }
            return parts.count == 2 ? PixelSize(width: parts[0], height: parts[1]) : nil
        }
        // Every advertised size must be one of the model's enumerated input shapes.
        let enumerated = Set(input.shapeConstraint.enumeratedShapes.compactMap { shape -> PixelSize? in
            shape.count == 4 ? PixelSize(width: shape[3].intValue, height: shape[2].intValue) : nil
        })
        let usable = enumerated.isEmpty ? sizes : sizes.filter(enumerated.contains)
        guard !usable.isEmpty else { throw SRError.incompatible("no usable input shapes") }
        var info = SuperResolutionModelInfo(
            id: id,
            family: user["ve.family"] ?? id,
            displayName: user["ve.displayName"] ?? id,
            architecture: user["ve.architecture"] ?? "unknown",
            kind: .coreML,
            scale: scale,
            license: description.metadata[.license] as? String ?? "unknown",
            source: user["ve.sourceRepo"].map { "huggingface.co/\($0)" } ?? "unknown",
            version: description.metadata[.versionString] as? String ?? "unknown",
            temporalFrames: user["ve.temporalFrames"].flatMap(Int.init) ?? 1,
            parameters: user["ve.parameters"].flatMap(Int.init),
            inputSizes: usable
        )
        info.hasFlexibleShapes = input.shapeConstraint.type == .range
        return info
    }

    nonisolated private static func contentHash(_ url: URL) -> String {
        var hasher = SHA256()
        let files = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey])
        while let file = files?.nextObject() as? URL {
            let values = try? file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            hasher.update(data: Data("\(file.lastPathComponent)|\(values?.fileSize ?? 0)|\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0)".utf8))
        }
        return hasher.finalize().prefix(6).map { String(format: "%02x", $0) }.joined()
    }

    nonisolated private static func directorySize(_ url: URL) -> Int {
        let files = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey])
        var total = 0
        while let file = files?.nextObject() as? URL {
            total += (try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        }
        return total
    }

    // MARK: - Lookup

    func entry(id: String) -> Entry? { entries.first { $0.id == id } }

    /// The variant of `family` whose scale best matches `preferredScale` (exact, else nearest). CPU-only
    /// needs a flexible-shape ("-flex") variant: Core ML's CPU backend aborts on enumerated input shapes
    /// (measured, Docs/02); the other compute units use the enumerated (Neural Engine friendly) ones.
    func variant(family: String, preferredScale: Int, units: ComputeUnitsOption = .cpuAndNeuralEngine) -> Entry? {
        let flexible = units == .cpuOnly
        let candidates = entries.filter { $0.info.family == family && $0.info.hasFlexibleShapes == flexible }
        return candidates.min { abs($0.info.scale - preferredScale) < abs($1.info.scale - preferredScale) }
    }

    func variants(family: String) -> [Entry] { entries.filter { $0.info.family == family } }

    /// Families that have a variant usable with `units`.
    func families(for units: ComputeUnitsOption) -> [(family: String, name: String)] {
        families.filter { variant(family: $0.family, preferredScale: 2, units: units) != nil }
    }

    var families: [(family: String, name: String)] {
        var seen = Set<String>()
        return entries.compactMap { entry in
            seen.insert(entry.info.family).inserted ? (entry.info.family, entry.info.displayName) : nil
        }
    }

    // MARK: - Loading

    /// Loads the model once for the given compute units; concurrent callers share the same load.
    func load(id: String, units: ComputeUnitsOption) async throws -> CoreMLSuperResolution {
        let key = "\(id)|\(units.rawValue)"
        if let model = loaded[key] { return model }
        if let task = loadTasks[key] { return try await task.value }
        guard let entry = entry(id: id) else { throw SRError.modelMissing(id) }
        if units == .cpuOnly, !entry.info.hasFlexibleShapes {
            // Measured: Core ML terminates the process (SIGTRAP) when an ML Program with enumerated
            // input shapes runs on the CPU only. Refuse instead of crashing.
            throw SRError.incompatible("CPU only needs a flexible-shape (-flex) model; \(entry.id) uses enumerated shapes")
        }

        update(id) { $0.state = .loading }
        let task = Task.detached(priority: .userInitiated) { () throws -> CoreMLSuperResolution in
            let configuration = MLModelConfiguration()
            configuration.computeUnits = units.mlComputeUnits
            configuration.allowLowPrecisionAccumulationOnGPU = true
            let model = try await MLModel.load(contentsOf: entry.compiledURL, configuration: configuration)
            return CoreMLSuperResolution(model: model, info: entry.info, computeUnits: units)
        }
        loadTasks[key] = task
        let start = CFAbsoluteTimeGetCurrent()
        do {
            let model = try await task.value
            loaded[key] = model
            loadTasks[key] = nil
            update(id) {
                $0.state = .loaded(units)
                $0.loadSeconds = CFAbsoluteTimeGetCurrent() - start
            }
            return model
        } catch {
            loadTasks[key] = nil
            update(id) { $0.state = .failed(error.localizedDescription) }
            throw error
        }
    }

    func unload(id: String) {
        loaded = loaded.filter { !$0.key.hasPrefix(id + "|") }
        update(id) { $0.state = .available; $0.warmUpSeconds = [:] }
    }

    func unloadAll() {
        loaded.removeAll()
        for entry in entries { update(entry.id) { $0.state = .available; $0.warmUpSeconds = [:] } }
    }

    /// Runs one prediction per size so the per-shape specialization (ANE compilation) happens
    /// before playback needs it. Records how long each first prediction took.
    func warmUp(_ model: CoreMLSuperResolution, sizes: [PixelSize]) async {
        for size in sizes where entry(id: model.info.id)?.warmUpSeconds[size] == nil {
            guard let buffer = try? model.makeInputBuffer(size: size) else { continue }
            CVPixelBufferLockBaseAddress(buffer, [])
            memset(CVPixelBufferGetBaseAddress(buffer), 0, CVPixelBufferGetDataSize(buffer))
            CVPixelBufferUnlockBaseAddress(buffer, [])
            let start = CFAbsoluteTimeGetCurrent()
            let context = SRFrameContext(sequence: 0, presentationTime: .zero, isDiscontinuity: true)
            if (try? await model.process(frame: buffer, context: context)) != nil {
                let seconds = CFAbsoluteTimeGetCurrent() - start
                update(model.info.id) { $0.warmUpSeconds[size] = seconds }
            }
        }
    }

    /// Where Core ML plans to run each operation (public `MLComputePlan` API). This is a plan,
    /// not a utilization measurement — the system does not expose Neural Engine utilization.
    func computePlan(id: String, units: ComputeUnitsOption) async -> ComputePlanSummary? {
        guard let entry = entry(id: id) else { return nil }
        let summary = await Task.detached(priority: .utility) { () -> ComputePlanSummary? in
            let configuration = MLModelConfiguration()
            configuration.computeUnits = units.mlComputeUnits
            guard let plan = try? await MLComputePlan.load(contentsOf: entry.compiledURL, configuration: configuration),
                  case .program(let program) = plan.modelStructure,
                  let main = program.functions["main"] else { return nil }
            var summary = ComputePlanSummary()
            var totalCost = 0.0, aneCost = 0.0
            func visit(_ block: MLModelStructure.Program.Block) {
                for operation in block.operations {
                    for nested in operation.blocks { visit(nested) }
                    guard let usage = plan.deviceUsage(for: operation) else { continue }
                    summary.operations += 1
                    let cost = plan.estimatedCost(of: operation)?.weight ?? 0
                    totalCost += cost
                    switch usage.preferred {
                    case .neuralEngine: summary.neuralEngine += 1; aneCost += cost
                    case .gpu: summary.gpu += 1
                    case .cpu: summary.cpu += 1
                    @unknown default: break
                    }
                }
            }
            visit(main.block)
            summary.neuralEngineCostShare = totalCost > 0 ? aneCost / totalCost : nil
            return summary
        }.value
        update(id) { $0.computePlan = summary }
        return summary
    }

    private func update(_ id: String, _ change: (inout Entry) -> Void) {
        guard let index = entries.firstIndex(where: { $0.id == id }) else { return }
        change(&entries[index])
    }
}
