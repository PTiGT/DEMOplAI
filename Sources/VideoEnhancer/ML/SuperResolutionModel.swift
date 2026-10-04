import CoreMedia
import CoreVideo
import Foundation

struct PixelSize: Hashable, Sendable, Codable, CustomStringConvertible {
    var width: Int
    var height: Int

    var description: String { "\(width)×\(height)" }
    var aspect: Double { Double(width) / Double(max(height, 1)) }
    var pixels: Int { width * height }

    func scaled(by factor: Int) -> PixelSize { PixelSize(width: width * factor, height: height * factor) }
}

/// Memory layout of the buffers a model reads and writes. The processing pipeline prepares the input
/// and decodes the output with Metal kernels matching the layout, so new backends only declare it.
enum SRBufferLayout: Sendable {
    /// `kCVPixelFormatType_OneComponent16Half`, width W, height 3·H: planar R, G, B (NCHW), values 0…1.
    case planarRGBFloat16
    /// `kCVPixelFormatType_64RGBAHalf`, interleaved RGBA, values 0…1.
    case rgbaFloat16
}

struct SuperResolutionModelInfo: Sendable, Equatable, Identifiable {
    enum Kind: String, Sendable {
        case coreML = "Core ML"
        case metal = "Metal (classical)"
    }

    var id: String
    var family: String
    var displayName: String
    var architecture: String
    var kind: Kind
    var scale: Int
    var license: String
    var source: String
    var version: String
    /// 1 = image SR applied per frame; > 1 = video SR that consumes neighbouring frames.
    var temporalFrames: Int
    var parameters: Int?
    var inputSizes: [PixelSize]
    /// Enumerated input shapes (Neural Engine friendly) or a flexible range (the CPU-only variants).
    var hasFlexibleShapes = false

    var isAI: Bool { kind == .coreML }

    /// "Real-ESRGAN General v3 ×2" (+ " · flexible shapes" for the CPU-only variants).
    var label: String { "\(displayName) ×\(scale)" + (hasFlexibleShapes ? " · flexible shapes (CPU only)" : "") }
}

/// Per-frame context. Image SR ignores most of it; video SR models use the sequence / discontinuity
/// information to manage their frame history.
struct SRFrameContext: Sendable {
    var sequence: UInt64
    var presentationTime: CMTime
    /// First frame after a seek, item change or scene cut: temporal state must be reset.
    var isDiscontinuity: Bool
}

/// A super-resolution backend. Implementations: `CoreMLSuperResolution` (Hugging Face models converted
/// to Core ML) and `MetalSuperResolution` (classical Lanczos + sharpening, for benchmarking against AI).
/// A multi-frame video SR model plugs in the same way with `temporalFrames > 1`.
protocol SuperResolutionModel: AnyObject, Sendable {
    var info: SuperResolutionModelInfo { get }
    var inputLayout: SRBufferLayout { get }
    var outputLayout: SRBufferLayout { get }

    /// A pooled, IOSurface-backed buffer to fill with the model input for `size` (one of `info.inputSizes`).
    func makeInputBuffer(size: PixelSize) throws -> CVPixelBuffer

    /// Runs the model. `frame` is laid out as `inputLayout`; the result as `outputLayout`,
    /// `info.scale` times larger.
    func process(frame: CVPixelBuffer, context: SRFrameContext) async throws -> CVPixelBuffer
}

enum SRError: LocalizedError {
    case bufferAllocation(CVReturn)
    case unsupportedInputSize(PixelSize)
    case modelMissing(String)
    case incompatible(String)

    var errorDescription: String? {
        switch self {
        case .bufferAllocation(let status): "Pixel buffer allocation failed (\(status))."
        case .unsupportedInputSize(let size): "The model has no input shape \(size)."
        case .modelMissing(let id): "Model \(id) is not installed."
        case .incompatible(let reason): "Incompatible model: \(reason)"
        }
    }
}

/// IOSurface-backed, Metal-compatible pixel buffer pools keyed by size.
final class PixelBufferPools: @unchecked Sendable {
    private let pixelFormat: OSType
    private let lock = NSLock()
    private var pools: [PixelSize: CVPixelBufferPool] = [:]

    init(pixelFormat: OSType) {
        self.pixelFormat = pixelFormat
    }

    /// For planar tensors pass the buffer size (width W, height 3·H), not the image size.
    func make(width: Int, height: Int) throws -> CVPixelBuffer {
        let key = PixelSize(width: width, height: height)
        let pool: CVPixelBufferPool = try lock.withLock {
            if let existing = pools[key] { return existing }
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            var created: CVPixelBufferPool?
            let status = CVPixelBufferPoolCreate(nil, [kCVPixelBufferPoolMinimumBufferCountKey: 3] as CFDictionary,
                                                 attributes as CFDictionary, &created)
            guard status == kCVReturnSuccess, let created else { throw SRError.bufferAllocation(status) }
            pools[key] = created
            return created
        }
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        guard status == kCVReturnSuccess, let buffer else { throw SRError.bufferAllocation(status) }
        return buffer
    }
}
