import CoreGraphics
import Metal

/// Classical GPU upscaler (Lanczos-2 with anti-ringing). Not AI: it only interpolates,
/// it cannot recover detail that is not in the source.
final class MetalUpscaler: @unchecked Sendable {
    private let context: MetalContext
    private let pool: TexturePool
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext, pool: TexturePool) {
        self.context = context
        self.pool = pool
        pipeline = context.computePipeline("upscaleLanczos")
    }

    func encode(_ input: MTLTexture, to size: (width: Int, height: Int), into commandBuffer: MTLCommandBuffer) -> PooledTexture {
        let output = pool.make(width: size.width, height: size.height, label: "Lanczos")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return output }
        encoder.label = "Upscale Lanczos-2"
        encoder.setTexture(input, index: 0)
        encoder.setTexture(output.texture, index: 1)
        context.dispatch(encoder, pipeline: pipeline, size: size)
        encoder.endEncoding()
        return output
    }

    /// Output size for upscaling `source` to fill `target` (aspect preserved). Never smaller than the
    /// source: downscaling to the window is left to the final present pass.
    static func outputSize(source: CGSize, target: CGSize, rotation: Int) -> (width: Int, height: Int) {
        // Work in the texture's (unrotated) orientation.
        let fitTarget = rotation % 180 == 0 ? target : CGSize(width: target.height, height: target.width)
        let rect = AspectFit.rect(content: source, in: fitTarget)
        let width = max(Int(source.width), Int(rect.width))
        let height = max(Int(source.height), Int(rect.height))
        return (width, height)
    }
}
