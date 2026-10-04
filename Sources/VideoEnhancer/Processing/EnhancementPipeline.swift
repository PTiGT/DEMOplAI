import CoreGraphics
import Metal

/// Must match `EnhanceParams` in Enhance.metal.
struct EnhanceParameters {
    var values: SIMD4<Float>
}

/// "Fast" mode: classical GPU enhancement, a few hundred microseconds per frame on Apple Silicon.
///   1. artifact reduction (edge-preserving bilateral) at source resolution
///   2. Lanczos-2 upscale to the display size
///   3. contrast-adaptive sharpening at display resolution
/// Also used to post-process AI output (sharpening only) and as the AI fallback.
final class EnhancementPipeline: @unchecked Sendable {
    private let context: MetalContext
    private let pool: TexturePool
    private let upscaler: MetalUpscaler
    private let denoisePipeline: MTLComputePipelineState
    private let sharpenPipeline: MTLComputePipelineState

    init(context: MetalContext, pool: TexturePool) {
        self.context = context
        self.pool = pool
        upscaler = MetalUpscaler(context: context, pool: pool)
        denoisePipeline = context.computePipeline("reduceArtifacts")
        sharpenPipeline = context.computePipeline("sharpenAdaptive")
    }

    func encode(_ source: PooledTexture, targetSize: CGSize, rotation: Int, settings: EnhancementSettings,
                into commandBuffer: MTLCommandBuffer) -> PooledTexture {
        var params = EnhanceParameters(values: SIMD4(settings.denoise, settings.sharpness, 0, 0))
        var current = source

        if settings.denoise > 0.001 {
            current = filter(current, pipeline: denoisePipeline, params: &params, label: "Reduce Artifacts", into: commandBuffer)
        }

        let size = MetalUpscaler.outputSize(source: CGSize(width: source.width, height: source.height),
                                            target: targetSize, rotation: rotation)
        if size.width > current.width || size.height > current.height {
            current = upscaler.encode(current.texture, to: size, into: commandBuffer)
        }

        if settings.sharpness > 0.001 {
            current = filter(current, pipeline: sharpenPipeline, params: &params, label: "Sharpen", into: commandBuffer)
        }
        return current
    }

    /// Sharpening only, at the texture's own size (used on AI output).
    func sharpen(_ input: PooledTexture, amount: Float, into commandBuffer: MTLCommandBuffer) -> PooledTexture {
        guard amount > 0.001 else { return input }
        var params = EnhanceParameters(values: SIMD4(0, amount, 0, 0))
        return filter(input, pipeline: sharpenPipeline, params: &params, label: "Sharpen", into: commandBuffer)
    }

    private func filter(_ input: PooledTexture, pipeline: MTLComputePipelineState, params: inout EnhanceParameters,
                        label: String, into commandBuffer: MTLCommandBuffer) -> PooledTexture {
        let output = pool.make(width: input.width, height: input.height, label: label)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return input }
        encoder.label = label
        encoder.setTexture(input.texture, index: 0)
        encoder.setTexture(output.texture, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<EnhanceParameters>.stride, index: 0)
        context.dispatch(encoder, pipeline: pipeline, size: (output.width, output.height))
        encoder.endEncoding()
        return output
    }
}
