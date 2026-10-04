import CoreMedia
import CoreVideo
import Metal
import QuartzCore

/// Must match `TensorParams` in ML.metal.
struct TensorParameters {
    var size: SIMD4<Int32>
    var values: SIMD4<Float>
}

/// A decoded frame after color conversion — the common input of every processing path.
final class SourceFrame: @unchecked Sendable {
    let rgb: PooledTexture
    let sequence: UInt64
    let presentationTime: CMTime
    let pulledAt: CFTimeInterval
    let dueAt: CFTimeInterval
    let rotation: Int
    let epoch: Int
    let nominalFrameRate: Double

    init(rgb: PooledTexture, sequence: UInt64, presentationTime: CMTime, pulledAt: CFTimeInterval,
         dueAt: CFTimeInterval, rotation: Int, epoch: Int, nominalFrameRate: Double) {
        self.rgb = rgb
        self.sequence = sequence
        self.presentationTime = presentationTime
        self.pulledAt = pulledAt
        self.dueAt = dueAt
        self.rotation = rotation
        self.epoch = epoch
        self.nominalFrameRate = nominalFrameRate
    }

    var size: PixelSize { PixelSize(width: rgb.width, height: rgb.height) }

    /// Same frame, due now (re-processing a paused frame after a settings change).
    func redue(at time: CFTimeInterval) -> SourceFrame {
        SourceFrame(rgb: rgb, sequence: sequence, presentationTime: presentationTime, pulledAt: time, dueAt: time,
                    rotation: rotation, epoch: epoch, nominalFrameRate: nominalFrameRate)
    }

    func frame(enhanced: PooledTexture?, path: EnhancementPath) -> ProcessedFrame {
        ProcessedFrame(sequence: sequence, presentationTime: presentationTime, pulledAt: pulledAt, dueAt: dueAt,
                       original: rgb, enhanced: enhanced, path: path, rotation: rotation, epoch: epoch)
    }
}

struct AIPostSettings: Codable, Hashable, Sendable {
    var consistency = true
    var stabilization = true
    var sharpen: Float = 0
}

/// Wraps a Core Video buffer so it can travel across concurrency domains; the pipeline never
/// mutates a buffer after handing it on.
struct SendableBuffer: @unchecked Sendable {
    let buffer: CVPixelBuffer
}

/// Runs a `SuperResolutionModel` asynchronously: Metal preprocessing → model → Metal post-processing.
/// At most `maxInFlight` frames are in the pipeline; callers decide what to do with frames that do
/// not fit (the engine keeps only the newest — backpressure).
final class SuperResolutionProcessor: @unchecked Sendable {
    /// Frames allowed in the pipeline at once. 2 overlaps GPU pre/post-processing of one frame with
    /// inference of another; whether that helps is measured by the benchmark (Docs/03).
    var maxFramesInFlight: Int {
        get { lock.withLock { maxInFlight } }
        set { lock.withLock { maxInFlight = max(1, newValue) } }
    }
    private var maxInFlight = 2

    private let context: MetalContext
    private let pool: TexturePool
    private let enhancement: EnhancementPipeline
    private let monitor: PerformanceMonitor
    private let toTensor: MTLComputePipelineState
    private let toRGBA: MTLComputePipelineState
    private let fromTensor: MTLComputePipelineState
    private let residualPipeline: MTLComputePipelineState
    private let applyPipeline: MTLComputePipelineState
    private let stabilizePipeline: MTLComputePipelineState
    private let blurPipeline: MTLComputePipelineState
    private let stillnessPipeline: MTLComputePipelineState

    private struct History {
        let epoch: Int
        let sequence: UInt64
        let modelID: String
        let output: PooledTexture
        let source: PooledTexture
    }

    private let lock = NSLock()
    private var inFlight = 0
    private var history: History?

    init(context: MetalContext, pool: TexturePool, enhancement: EnhancementPipeline, monitor: PerformanceMonitor) {
        self.context = context
        self.pool = pool
        self.enhancement = enhancement
        self.monitor = monitor
        toTensor = context.computePipeline("rgbToPlanarTensor")
        toRGBA = context.computePipeline("rgbToRGBA")
        fromTensor = context.computePipeline("planarTensorToRGB")
        residualPipeline = context.computePipeline("consistencyResidual")
        applyPipeline = context.computePipeline("consistencyApply")
        stabilizePipeline = context.computePipeline("temporalStabilize")
        blurPipeline = context.computePipeline("blurResidual")
        stillnessPipeline = context.computePipeline("stillnessMask")
    }

    var isAtCapacity: Bool { lock.withLock { inFlight >= maxInFlight } }
    var framesInFlight: Int { lock.withLock { inFlight } }

    /// Drops temporal history (seek, new item, model change).
    func resetHistory() {
        lock.withLock { history = nil }
    }

    /// Input shape for a source: closest aspect ratio first, then the smallest height that keeps
    /// ≥ 95% of the source height (never throw away source resolution), else the largest.
    static func inputSize(for source: PixelSize, sizes: [PixelSize]) -> PixelSize? {
        guard !sizes.isEmpty else { return nil }
        func aspectError(_ s: PixelSize) -> Double { abs(log(s.aspect / source.aspect)) }
        let best = sizes.map(aspectError).min()!
        let candidates = sizes.filter { aspectError($0) <= best + 0.02 }
        if let fit = candidates.filter({ Double($0.height) >= Double(source.height) * 0.95 }).min(by: { $0.height < $1.height }) {
            return fit
        }
        return candidates.max { $0.height < $1.height }
    }

    /// Starts processing `source`. Returns false (and does nothing) when already at capacity.
    /// `completion` is called exactly once for accepted frames, on an arbitrary thread.
    func submit(_ source: SourceFrame, model: any SuperResolutionModel, settings: AIPostSettings,
                completion: @escaping @Sendable (Result<ProcessedFrame, Error>) -> Void) -> Bool {
        let accepted = lock.withLock {
            guard inFlight < maxInFlight else { return false }
            inFlight += 1
            return true
        }
        guard accepted else { return false }

        let start = CACurrentMediaTime()
        do {
            guard let size = Self.inputSize(for: source.size, sizes: model.info.inputSizes) else {
                throw SRError.unsupportedInputSize(source.size)
            }
            let input = try model.makeInputBuffer(size: size)
            let wrapped = try MetalTextureCache.wrap(input, cache: context.textureCache, usage: [.shaderRead, .shaderWrite])
            guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { throw SRError.bufferAllocation(-1) }
            commandBuffer.label = "AI preprocess"
            encoder.label = "Source → model input \(size)"
            encoder.setTexture(source.rgb.texture, index: 0)
            encoder.setTexture(wrapped.textures[0], index: 1)
            switch model.inputLayout {
            case .planarRGBFloat16:
                var params = TensorParameters(size: SIMD4(Int32(size.width), Int32(size.height), 0, 0), values: .zero)
                encoder.setBytes(&params, length: MemoryLayout<TensorParameters>.stride, index: 0)
                context.dispatch(encoder, pipeline: toTensor, size: (size.width, size.height))
            case .rgbaFloat16:
                context.dispatch(encoder, pipeline: toRGBA, size: (size.width, size.height))
            }
            encoder.endEncoding()

            let inputBuffer = SendableBuffer(buffer: input)
            commandBuffer.addCompletedHandler { [self] buffer in
                _ = wrapped
                let preprocessGPU = buffer.gpuEndTime - buffer.gpuStartTime
                monitor.recordGPU(start: buffer.gpuStartTime, end: buffer.gpuEndTime, isRenderPass: false)
                Task {
                    await self.infer(source: source, model: model, input: inputBuffer, inputSize: size, settings: settings,
                                     preprocessGPU: preprocessGPU, start: start, completion: completion)
                }
            }
            commandBuffer.commit()
        } catch {
            finish()
            completion(.failure(error))
        }
        return true
    }

    private func infer(source: SourceFrame, model: any SuperResolutionModel, input: SendableBuffer, inputSize: PixelSize,
                       settings: AIPostSettings, preprocessGPU: Double, start: CFTimeInterval,
                       completion: @escaping @Sendable (Result<ProcessedFrame, Error>) -> Void) async {
        let inferenceStart = CACurrentMediaTime()
        do {
            let context = SRFrameContext(sequence: source.sequence, presentationTime: source.presentationTime,
                                         isDiscontinuity: false)
            let output = try await model.process(frame: input.buffer, context: context)
            let inference = CACurrentMediaTime() - inferenceStart
            try postprocess(output: SendableBuffer(buffer: output), source: source, model: model, inputSize: inputSize,
                            settings: settings, preprocessGPU: preprocessGPU, inference: inference, start: start,
                            completion: completion)
        } catch {
            finish()
            completion(.failure(error))
        }
    }

    private func postprocess(output: SendableBuffer, source: SourceFrame, model: any SuperResolutionModel, inputSize: PixelSize,
                             settings: AIPostSettings, preprocessGPU: Double, inference: Double, start: CFTimeInterval,
                             completion: @escaping @Sendable (Result<ProcessedFrame, Error>) -> Void) throws {
        let planes = try MetalTextureCache.wrap(output.buffer, cache: context.textureCache)
        let outSize = inputSize.scaled(by: model.info.scale)
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { throw SRError.bufferAllocation(-1) }
        commandBuffer.label = "AI postprocess"

        var current = pool.make(width: outSize.width, height: outSize.height, label: "AI output")
        switch model.outputLayout {
        case .planarRGBFloat16:
            encode(fromTensor, inputs: [planes.textures[0]], output: current.texture, size: outSize, into: commandBuffer, label: "Model output → RGB")
        case .rgbaFloat16:
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.copy(from: planes.textures[0], to: current.texture)
                blit.endEncoding()
            }
        }

        if settings.consistency {
            let residual = pool.make(width: inputSize.width, height: inputSize.height, label: "Consistency residual")
            encode(residualPipeline, inputs: [source.rgb.texture, current.texture], output: residual.texture,
                   size: inputSize, into: commandBuffer, label: "Consistency residual")
            let blurred = pool.make(width: inputSize.width, height: inputSize.height, label: "Consistency residual blurred")
            encode(blurPipeline, inputs: [residual.texture], output: blurred.texture, size: inputSize,
                   into: commandBuffer, label: "Blur residual")
            let corrected = pool.make(width: outSize.width, height: outSize.height, label: "AI consistent")
            var params = TensorParameters(size: .zero, values: SIMD4(1, 0, 0, 0))
            encode(applyPipeline, inputs: [current.texture, blurred.texture], output: corrected.texture, size: outSize,
                   params: &params, into: commandBuffer, label: "Consistency apply")
            current = corrected
        }

        let previous: History? = lock.withLock { history }
        if settings.stabilization, let previous, previous.epoch == source.epoch, previous.modelID == model.info.id,
           previous.sequence < source.sequence, previous.output.width == outSize.width, previous.output.height == outSize.height,
           previous.source.width == source.rgb.width, previous.source.height == source.rgb.height {
            let mask = pool.make(width: source.rgb.width, height: source.rgb.height, format: .r16Float, label: "Stillness")
            encode(stillnessPipeline, inputs: [source.rgb.texture, previous.source.texture], output: mask.texture,
                   size: source.size, into: commandBuffer, label: "Stillness mask")
            let stabilized = pool.make(width: outSize.width, height: outSize.height, label: "AI stabilized")
            var params = TensorParameters(size: .zero, values: SIMD4(0.5, 0, 0, 0))
            encode(stabilizePipeline, inputs: [current.texture, previous.output.texture, mask.texture],
                   output: stabilized.texture, size: outSize, params: &params, into: commandBuffer, label: "Temporal stabilize")
            current = stabilized
        }

        if settings.sharpen > 0.001 {
            current = enhancement.sharpen(current, amount: settings.sharpen, into: commandBuffer)
        }

        lock.withLock {
            if history == nil || history!.epoch != source.epoch || history!.sequence < source.sequence {
                history = History(epoch: source.epoch, sequence: source.sequence, modelID: model.info.id,
                                  output: current, source: source.rgb)
            }
        }

        let result = current
        let path = EnhancementPath.ai(model: model.info.displayName)
        commandBuffer.addCompletedHandler { [self] buffer in
            _ = planes
            monitor.recordGPU(start: buffer.gpuStartTime, end: buffer.gpuEndTime, isRenderPass: false)
            monitor.recordAI(SRTimings(preprocessGPU: preprocessGPU, inference: inference,
                                       postprocessGPU: buffer.gpuEndTime - buffer.gpuStartTime,
                                       total: CACurrentMediaTime() - start, inputSize: inputSize, outputSize: outSize))
            finish()
            completion(.success(source.frame(enhanced: result, path: path)))
        }
        commandBuffer.commit()
    }

    private func encode(_ pipeline: MTLComputePipelineState, inputs: [MTLTexture], output: MTLTexture, size: PixelSize,
                        into commandBuffer: MTLCommandBuffer, label: String) {
        var params = TensorParameters(size: .zero, values: .zero)
        encode(pipeline, inputs: inputs, output: output, size: size, params: &params, into: commandBuffer, label: label)
    }

    private func encode(_ pipeline: MTLComputePipelineState, inputs: [MTLTexture], output: MTLTexture, size: PixelSize,
                        params: inout TensorParameters, into commandBuffer: MTLCommandBuffer, label: String) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = label
        for (index, texture) in inputs.enumerated() { encoder.setTexture(texture, index: index) }
        encoder.setTexture(output, index: inputs.count)
        encoder.setBytes(&params, length: MemoryLayout<TensorParameters>.stride, index: 0)
        context.dispatch(encoder, pipeline: pipeline, size: (size.width, size.height))
        encoder.endEncoding()
    }

    private func finish() {
        lock.withLock { inFlight -= 1 }
    }
}
