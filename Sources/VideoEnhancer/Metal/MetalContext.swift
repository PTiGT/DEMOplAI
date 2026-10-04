import CoreVideo
import Foundation
import Metal
import QuartzCore
import simd

/// Device, queue, shader library and pipeline states shared by the whole pipeline.
/// Immutable after creation; Metal device/queue/pipeline objects are thread-safe.
final class MetalContext: @unchecked Sendable {
    enum SetupError: LocalizedError {
        case noDevice
        case shadersMissing
        case textureCache(CVReturn)

        var errorDescription: String? {
            switch self {
            case .noDevice: "No Metal device available."
            case .shadersMissing: "Shader sources are missing from the app bundle."
            case .textureCache(let status): "CVMetalTextureCacheCreate failed (\(status))."
            }
        }
    }

    static let shared: Result<MetalContext, Error> = Result { try MetalContext() }

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    let textureCache: CVMetalTextureCache
    /// Time spent compiling shaders at startup (runtime compilation, see Package.swift).
    let shaderCompileTime: TimeInterval

    private let lock = NSLock()
    private var computePipelines: [String: MTLComputePipelineState] = [:]
    private var renderPipelines: [String: MTLRenderPipelineState] = [:]

    private init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw SetupError.noDevice
        }
        self.device = device
        self.commandQueue = queue
        queue.label = "VideoEnhancer"

        let start = CACurrentMediaTime()
        library = try device.makeLibrary(source: try Self.loadShaderSource(), options: Self.compileOptions)
        shaderCompileTime = CACurrentMediaTime() - start

        var cache: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        guard status == kCVReturnSuccess, let cache else { throw SetupError.textureCache(status) }
        textureCache = cache
    }

    private static var compileOptions: MTLCompileOptions {
        let options = MTLCompileOptions()
        options.mathMode = .fast
        options.languageVersion = .version3_1
        return options
    }

    /// Concatenates the `.metal` files, Common first so the others can use its declarations.
    private static func loadShaderSource() throws -> String {
        guard let folder = AppResources.bundle.url(forResource: "Shaders", withExtension: nil),
              let files = try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
        else { throw SetupError.shadersMissing }
        let sources = files
            .filter { $0.pathExtension == "metal" }
            .sorted { lhs, rhs in
                if lhs.lastPathComponent == "Common.metal" { return true }
                if rhs.lastPathComponent == "Common.metal" { return false }
                return lhs.lastPathComponent < rhs.lastPathComponent
            }
        guard !sources.isEmpty else { throw SetupError.shadersMissing }
        return try sources.map { "// ---- \($0.lastPathComponent)\n" + (try String(contentsOf: $0, encoding: .utf8)) }
            .joined(separator: "\n")
    }

    func computePipeline(_ name: String) -> MTLComputePipelineState {
        lock.withLock {
            if let cached = computePipelines[name] { return cached }
            guard let function = library.makeFunction(name: name) else {
                fatalError("Missing compute function \(name)")
            }
            let state = try! device.makeComputePipelineState(function: function)
            computePipelines[name] = state
            return state
        }
    }

    func renderPipeline(vertex: String, fragment: String, pixelFormat: MTLPixelFormat) -> MTLRenderPipelineState {
        let key = "\(vertex)|\(fragment)|\(pixelFormat.rawValue)"
        return lock.withLock {
            if let cached = renderPipelines[key] { return cached }
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = library.makeFunction(name: vertex)
            descriptor.fragmentFunction = library.makeFunction(name: fragment)
            descriptor.colorAttachments[0].pixelFormat = pixelFormat
            let state = try! device.makeRenderPipelineState(descriptor: descriptor)
            renderPipelines[key] = state
            return state
        }
    }

    /// Encodes a 2D compute dispatch covering `size` with one thread per pixel.
    func dispatch(_ encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, size: (width: Int, height: Int)) {
        encoder.setComputePipelineState(pipeline)
        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        encoder.dispatchThreads(
            MTLSize(width: size.width, height: size.height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1)
        )
    }
}
