import CoreML
import Foundation

enum QualityMode: String, CaseIterable, Codable, Sendable, Identifiable {
    case original, fast, ai, aiHighQuality

    var id: String { rawValue }

    var title: String {
        switch self {
        case .original: "Original"
        case .fast: "Fast"
        case .ai: "AI"
        case .aiHighQuality: "AI High Quality"
        }
    }

    var detail: String {
        switch self {
        case .original: "No processing"
        case .fast: "Metal: artifact reduction, Lanczos upscale, adaptive sharpening"
        case .ai: "Real-time AI super resolution; falls back to Fast when it cannot keep up"
        case .aiHighQuality: "Largest model at full input size, even if slower than real time"
        }
    }

    var usesAI: Bool { self == .ai || self == .aiHighQuality }
}

enum ComputeUnitsOption: String, CaseIterable, Codable, Sendable, Identifiable {
    case cpuAndNeuralEngine, all, cpuAndGPU, cpuOnly

    var id: String { rawValue }

    var title: String {
        switch self {
        case .cpuAndNeuralEngine: "CPU + Neural Engine"
        case .all: "All (CPU + GPU + Neural Engine)"
        case .cpuAndGPU: "CPU + GPU"
        case .cpuOnly: "CPU only"
        }
    }

    var shortTitle: String {
        switch self {
        case .cpuAndNeuralEngine: "CPU+ANE"
        case .all: "ALL"
        case .cpuAndGPU: "CPU+GPU"
        case .cpuOnly: "CPU"
        }
    }

    var mlComputeUnits: MLComputeUnits {
        switch self {
        case .cpuAndNeuralEngine: .cpuAndNeuralEngine
        case .all: .all
        case .cpuAndGPU: .cpuAndGPU
        case .cpuOnly: .cpuOnly
        }
    }
}

struct EnhancementSettings: Codable, Equatable, Sendable {
    /// Contrast-adaptive sharpening, 0…1.
    var sharpness: Float = 0.35
    /// Edge-preserving artifact reduction at source resolution, 0…1.
    var denoise: Float = 0.25
}

struct ProcessingConfiguration: Codable, Equatable, Sendable {
    var mode: QualityMode = .ai
    /// Automatically drop to a lighter path when the selected one cannot keep up.
    var dynamicQuality = true
    var enhancement = EnhancementSettings()
    /// Model family for real-time AI. SPAN ×2 is the fastest on the M4 Neural Engine (Docs/02).
    var aiModelFamily = "span-nomosuni"
    /// Model family for AI High Quality.
    var hqModelFamily = "realesr-general-x4v3"
    /// CPU+ANE and ALL measured equal and fastest; CPU+GPU 1.6–2.6× slower; CPU-only ~8 fps at 360p and
    /// needs the flexible-shape "-flex" model variants (Docs/02_MODEL_EVALUATION.md, Docs/03_PERFORMANCE.md).
    var computeUnits: ComputeUnitsOption = .cpuAndNeuralEngine
    /// Blend AI output with the previous output where the source did not change (less flicker).
    var temporalStabilization = true
    /// Pull low frequencies of the AI output back to the source (fixes color/brightness drift).
    var consistencyCorrection = true
    /// Extra adaptive sharpening on AI output, 0…1.
    var aiSharpen: Float = 0

    static let storageKey = "processing.configuration"

    static func load() -> ProcessingConfiguration {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let config = try? JSONDecoder().decode(ProcessingConfiguration.self, from: data) else {
            return ProcessingConfiguration()
        }
        return config
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: Self.storageKey)
        }
    }
}
