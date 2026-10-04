import AVFoundation
import CoreML
import Foundation
import Metal
import QuartzCore

struct BenchmarkConfiguration: Codable, Sendable, Hashable {
    /// Model id (e.g. "span-nomosuni-x2"); nil benchmarks the classical Metal Fast path.
    var modelID: String?
    var computeUnits: ComputeUnitsOption = .cpuAndNeuralEngine
    var inputSize = PixelSize(width: 1280, height: 720)
    /// Output size for the Metal path (AI output size is input × model scale).
    var metalOutputSize = PixelSize(width: 1920, height: 1080)
    var duration: Double = 10
    /// Frames arrive at this rate (a video playing in real time). Frames that arrive while the
    /// pipeline is full are dropped, exactly like during playback.
    var arrivalFPS: Double = 60
    /// 2 = same as playback (SuperResolutionProcessor default).
    var framesInFlight = 2
    /// Decode frames from this file; nil uses a synthetic frame.
    var sourceFile: URL?
    var postprocessing = AIPostSettings()
    /// Write the first processed frames (in order) as PNG here — pipeline verification.
    var dumpDirectory: URL?
    var dumpCount = 8
}

struct BenchmarkResult: Codable, Sendable {
    var configuration: BenchmarkConfiguration
    var modelName: String
    var outputSize: PixelSize
    var framesOffered = 0
    var framesProcessed = 0
    var framesDropped = 0
    var averageFPS = 0.0
    var inferenceMedianMs: Double?
    var inferenceP95Ms: Double?
    var inferenceMeanMs: Double?
    var totalLatencyMedianMs: Double?
    var preprocessGPUMs: Double?
    var postprocessGPUMs: Double?
    var peakMemoryBytes: UInt64 = 0
    var cpuPercent: Double?
    var gpuBusyPercent: Double?
    var loadSeconds: Double?
    var firstRunSeconds: Double?
    var error: String?

    /// Real time = every arriving frame processed.
    var isRealTime: Bool { framesOffered > 0 && Double(framesDropped) / Double(framesOffered) < 0.01 }
}

/// Runs the same processing code as playback (YCbCr conversion, SuperResolutionProcessor or the
/// Metal enhancement pipeline) on a stream of frames and measures it.
final class BenchmarkRunner: @unchecked Sendable {
    private let context: MetalContext
    private let pool: TexturePool

    init(context: MetalContext) {
        self.context = context
        pool = TexturePool(device: context.device)
    }

    func run(_ config: BenchmarkConfiguration, model: CoreMLSuperResolution?, loadSeconds: Double?,
             progress: (@Sendable (Double) -> Void)? = nil) async -> BenchmarkResult {
        var result = BenchmarkResult(configuration: config, modelName: model.map { $0.info.label } ?? "Metal Fast (Lanczos + CAS)",
                                     outputSize: model.map { config.inputSize.scaled(by: $0.info.scale) } ?? config.metalOutputSize)
        result.loadSeconds = loadSeconds
        do {
            let frames = try await loadSourceFrames(config)
            if let model {
                try await runAI(config, model: model, frames: frames, result: &result, progress: progress)
            } else {
                try runMetal(config, frames: frames, result: &result, progress: progress)
            }
        } catch {
            result.error = error.localizedDescription
        }
        return result
    }

    // MARK: - AI

    private func runAI(_ config: BenchmarkConfiguration, model: CoreMLSuperResolution, frames: [PooledTexture],
                       result: inout BenchmarkResult, progress: (@Sendable (Double) -> Void)?) async throws {
        let monitor = PerformanceMonitor()
        let processor = SuperResolutionProcessor(context: context, pool: pool,
                                                 enhancement: EnhancementPipeline(context: context, pool: pool), monitor: monitor)
        processor.maxFramesInFlight = config.framesInFlight

        // Warm-up: first prediction for this shape includes device-specific compilation.
        let warmStart = CACurrentMediaTime()
        _ = try await processOne(processor, frame: frames[0], sequence: 0, model: model, settings: config.postprocessing)
        result.firstRunSeconds = CACurrentMediaTime() - warmStart
        _ = try await processOne(processor, frame: frames[0], sequence: 1, model: model, settings: config.postprocessing)
        monitor.reset()

        if let directory = config.dumpDirectory {
            try await dumpFrames(processor: processor, frames: frames, model: model, config: config, directory: directory)
        }

        let counter = Counter()
        var cpu = CPUUsageSampler()
        _ = cpu.sample()
        let start = CACurrentMediaTime()
        let interval = 1 / config.arrivalFPS
        var offered = 0
        var peak: UInt64 = 0
        while CACurrentMediaTime() - start < config.duration {
            let due = start + Double(offered) * interval
            let wait = due - CACurrentMediaTime()
            if wait > 0 { try await Task.sleep(nanoseconds: UInt64(wait * 1_000_000_000)) }
            let frame = frames[offered % frames.count]
            let source = SourceFrame(rgb: frame, sequence: UInt64(offered + 2), presentationTime: .zero, pulledAt: due,
                                     dueAt: due, rotation: 0, epoch: 0, nominalFrameRate: config.arrivalFPS)
            let accepted = processor.submit(source, model: model, settings: config.postprocessing) { outcome in
                if case .success = outcome { counter.increment() } else { counter.fail() }
            }
            if !accepted { result.framesDropped += 1 }
            offered += 1
            if offered % 15 == 0 {
                peak = max(peak, SystemMetrics.memoryFootprintBytes() ?? 0)
                progress?(min(1, (CACurrentMediaTime() - start) / config.duration))
            }
        }
        let elapsed = CACurrentMediaTime() - start
        while processor.framesInFlight > 0 { try await Task.sleep(nanoseconds: 2_000_000) }

        let snapshot = monitor.snapshot()
        let stats = monitor.aiStatistics
        result.framesOffered = offered
        result.framesProcessed = counter.value
        result.averageFPS = Double(counter.value) / elapsed
        result.inferenceMedianMs = snapshot.inferenceMs
        result.inferenceP95Ms = snapshot.inferenceP95Ms
        result.inferenceMeanMs = monitor.inferenceMean
        result.totalLatencyMedianMs = stats.latencyMedian
        result.preprocessGPUMs = snapshot.preprocessGPUMs
        result.postprocessGPUMs = snapshot.postprocessGPUMs
        result.peakMemoryBytes = max(peak, SystemMetrics.memoryFootprintBytes() ?? 0)
        result.cpuPercent = cpu.sample()
        result.gpuBusyPercent = monitor.gpuBusyPercent(over: elapsed)
        if counter.failures > 0 { result.error = "\(counter.failures) frames failed" }
    }

    /// Processes the first frames strictly in order (so temporal stabilization sees a real sequence)
    /// and saves source and result side by side as PNG files.
    private func dumpFrames(processor: SuperResolutionProcessor, frames: [PooledTexture], model: CoreMLSuperResolution,
                            config: BenchmarkConfiguration, directory: URL) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        processor.resetHistory()
        for index in 0..<min(config.dumpCount, frames.count) {
            let result = try await processOne(processor, frame: frames[index], sequence: UInt64(1000 + index), model: model,
                                              settings: config.postprocessing)
            let name = String(format: "%02d", index)
            try await TextureExport.writePNG(result.enhanced!.texture, to: directory.appendingPathComponent("\(name)-enhanced.png"), context: context)
            try await TextureExport.writePNG(frames[index].texture, to: directory.appendingPathComponent("\(name)-source.png"), context: context)
        }
        processor.resetHistory()
    }

    private func processOne(_ processor: SuperResolutionProcessor, frame: PooledTexture, sequence: UInt64,
                            model: CoreMLSuperResolution, settings: AIPostSettings) async throws -> ProcessedFrame {
        let source = SourceFrame(rgb: frame, sequence: sequence, presentationTime: .zero, pulledAt: 0, dueAt: 0,
                                 rotation: 0, epoch: 0, nominalFrameRate: 0)
        return try await withCheckedThrowingContinuation { continuation in
            let accepted = processor.submit(source, model: model, settings: settings) { continuation.resume(with: $0) }
            if !accepted { continuation.resume(throwing: SRError.bufferAllocation(-2)) }
        }
    }

    // MARK: - Metal Fast

    private func runMetal(_ config: BenchmarkConfiguration, frames: [PooledTexture], result: inout BenchmarkResult,
                          progress: (@Sendable (Double) -> Void)?) throws {
        let enhancement = EnhancementPipeline(context: context, pool: pool)
        let monitor = PerformanceMonitor()
        let target = CGSize(width: config.metalOutputSize.width, height: config.metalOutputSize.height)
        var cpu = CPUUsageSampler()
        _ = cpu.sample()
        let start = CACurrentMediaTime()
        var processed = 0
        var gpuTimes = SampleWindow(capacity: 100_000)
        while CACurrentMediaTime() - start < config.duration {
            guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { break }
            let frame = frames[processed % frames.count]
            let output = enhancement.encode(frame, targetSize: target, rotation: 0, settings: EnhancementSettings(), into: commandBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            _ = output
            gpuTimes.add((commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000)
            monitor.recordGPU(start: commandBuffer.gpuStartTime, end: commandBuffer.gpuEndTime, isRenderPass: true)
            processed += 1
            if processed % 30 == 0 { progress?(min(1, (CACurrentMediaTime() - start) / config.duration)) }
        }
        let elapsed = CACurrentMediaTime() - start
        result.framesOffered = processed
        result.framesProcessed = processed
        result.averageFPS = Double(processed) / elapsed
        result.inferenceMedianMs = gpuTimes.median
        result.inferenceP95Ms = gpuTimes.p95
        result.inferenceMeanMs = gpuTimes.mean
        result.peakMemoryBytes = SystemMetrics.memoryFootprintBytes() ?? 0
        result.cpuPercent = cpu.sample()
        result.gpuBusyPercent = monitor.gpuBusyPercent(over: elapsed)
    }

    // MARK: - Source frames

    /// Up to 30 frames at the benchmark input size (decoded from a file, or one synthetic frame).
    private func loadSourceFrames(_ config: BenchmarkConfiguration) async throws -> [PooledTexture] {
        let resample = context.computePipeline("rgbToRGBA")
        let converter = YCbCrConverter(context: context, pool: pool)
        var frames: [PooledTexture] = []

        if let url = config.sourceFile {
            let asset = AVURLAsset(url: url)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw VideoInfo.LoadError.noVideoTrack }
            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ])
            reader.add(output)
            reader.startReading()
            while frames.count < 30, let sample = output.copyNextSampleBuffer() {
                guard let buffer = CMSampleBufferGetImageBuffer(sample) else { continue }
                let frame = VideoFrame(pixelBuffer: buffer, presentationTime: sample.presentationTimeStamp, targetHostTime: 0,
                                       sequenceNumber: UInt64(frames.count), rotation: 0, isDiscontinuity: false, nominalFrameRate: 0)
                guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { break }
                let rgb = try converter.convert(frame, into: commandBuffer)
                frames.append(resampled(rgb.texture, config.inputSize, pipeline: resample, commandBuffer: commandBuffer))
                commandBuffer.commit()
                await commandBuffer.completed()
            }
            reader.cancelReading()
        }
        if frames.isEmpty {
            frames.append(syntheticFrame(config.inputSize))
        }
        return frames
    }

    private func resampled(_ texture: MTLTexture, _ size: PixelSize, pipeline: MTLComputePipelineState,
                           commandBuffer: MTLCommandBuffer) -> PooledTexture {
        let output = pool.make(width: size.width, height: size.height, label: "Benchmark frame")
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setTexture(texture, index: 0)
        encoder.setTexture(output.texture, index: 1)
        context.dispatch(encoder, pipeline: pipeline, size: (size.width, size.height))
        encoder.endEncoding()
        return output
    }

    /// Deterministic detailed pattern (gradients + fine stripes) uploaded through a shared buffer.
    private func syntheticFrame(_ size: PixelSize) -> PooledTexture {
        let texture = pool.make(width: size.width, height: size.height, label: "Synthetic")
        var pixels = [Float16](repeating: 1, count: size.width * size.height * 4)
        for y in 0..<size.height {
            for x in 0..<size.width {
                let i = (y * size.width + x) * 4
                let stripes: Float = ((x / 2 + y / 3) % 5 == 0) ? 0.3 : 0
                pixels[i] = Float16(Float(x) / Float(size.width) * 0.7 + stripes)
                pixels[i + 1] = Float16(Float(y) / Float(size.height) * 0.6 + stripes)
                pixels[i + 2] = Float16(0.4 + 0.3 * sin(Float(x + y) * 0.05))
            }
        }
        let staging = context.device.makeTexture(descriptor: {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: size.width, height: size.height, mipmapped: false)
            d.storageMode = .shared
            return d
        }())!
        staging.replace(region: MTLRegionMake2D(0, 0, size.width, size.height), mipmapLevel: 0, withBytes: pixels,
                        bytesPerRow: size.width * 8)
        let commandBuffer = context.commandQueue.makeCommandBuffer()!
        let blit = commandBuffer.makeBlitCommandEncoder()!
        blit.copy(from: staging, to: texture.texture)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return texture
    }
}

/// Thread-safe completion counter for benchmark callbacks.
private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    private var failed = 0
    func increment() { lock.withLock { count += 1 } }
    func fail() { lock.withLock { failed += 1 } }
    var value: Int { lock.withLock { count } }
    var failures: Int { lock.withLock { failed } }
}
