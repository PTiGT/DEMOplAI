import CoreVideo
import Metal
import simd

/// Must match `YCbCrParams` in Common.metal.
struct YCbCrParameters {
    var toRGB: simd_float4x4
    var scale: SIMD4<Float>
    var offset: SIMD4<Float>
    var flags: SIMD4<Int32>
}

/// Converts decoded 4:2:0 frames into the working RGB format (rgba16Float, BT.709 gamma-encoded)
/// on the GPU. Everything downstream (enhancement, AI, comparison, display) starts from this texture.
final class YCbCrConverter: @unchecked Sendable {
    private let context: MetalContext
    private let pool: TexturePool
    private let yuvPipeline: MTLComputePipelineState
    private let bgraPipeline: MTLComputePipelineState

    init(context: MetalContext, pool: TexturePool) {
        self.context = context
        self.pool = pool
        yuvPipeline = context.computePipeline("convertYCbCr420")
        bgraPipeline = context.computePipeline("convertBGRA")
    }

    /// Encodes the conversion; the source planes stay alive until `commandBuffer` completes.
    func convert(_ frame: VideoFrame, into commandBuffer: MTLCommandBuffer) throws -> PooledTexture {
        let planes = try MetalTextureCache.wrap(frame.pixelBuffer, cache: context.textureCache)
        let output = pool.make(width: frame.width, height: frame.height, label: "Source RGB")

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return output }
        encoder.label = "Convert \(PixelFormat.fourCC(frame.pixelFormat))"
        if planes.textures.count == 2 {
            var params = Self.parameters(for: frame.pixelBuffer)
            encoder.setTexture(planes.textures[0], index: 0)
            encoder.setTexture(planes.textures[1], index: 1)
            encoder.setTexture(output.texture, index: 2)
            encoder.setBytes(&params, length: MemoryLayout<YCbCrParameters>.stride, index: 0)
            context.dispatch(encoder, pipeline: yuvPipeline, size: (output.width, output.height))
        } else {
            encoder.setTexture(planes.textures[0], index: 0)
            encoder.setTexture(output.texture, index: 2)
            context.dispatch(encoder, pipeline: bgraPipeline, size: (output.width, output.height))
        }
        encoder.endEncoding()
        commandBuffer.addCompletedHandler { _ in _ = planes }
        return output
    }

    /// (Kr, Kb) per YCbCr matrix.
    private static let lumaCoefficients: [String: (Float, Float)] = [
        kCVImageBufferYCbCrMatrix_ITU_R_709_2 as String: (0.2126, 0.0722),
        kCVImageBufferYCbCrMatrix_ITU_R_601_4 as String: (0.299, 0.114),
        kCVImageBufferYCbCrMatrix_ITU_R_2020 as String: (0.2627, 0.0593),
        kCVImageBufferYCbCrMatrix_SMPTE_240M_1995 as String: (0.212, 0.087),
    ]

    /// Range expansion and matrix from the buffer's format and color attachments.
    static func parameters(for buffer: CVPixelBuffer) -> YCbCrParameters {
        let format = CVPixelBufferGetPixelFormatType(buffer)
        let color = ColorInfo(pixelBuffer: buffer)

        let tenBit = [
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
        ].contains(format)
        let fullRange = [
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
            kCVPixelFormatType_Lossless_420YpCbCr8BiPlanarFullRange,
        ].contains(format)

        // Normalized texture value → integer code value. 10-bit samples are MSB-aligned in 16 bits.
        let codeScale: Float = tenBit ? 65535.0 / 64.0 : 255.0
        let (yBlack, yRange, cZero, cRange): (Float, Float, Float, Float) = switch (tenBit, fullRange) {
        case (false, false): (16, 219, 128, 224)
        case (false, true): (0, 255, 128, 255)
        case (true, false): (64, 876, 512, 896)
        case (true, true): (0, 1023, 512, 1023)
        }
        let scale = SIMD4<Float>(codeScale / yRange, codeScale / cRange, codeScale / cRange, 0)
        let offset = SIMD4<Float>(-yBlack / yRange, -cZero / cRange, -cZero / cRange, 0)

        // Untagged: the usual convention is BT.601 for SD and BT.709 for HD.
        let (kr, kb) = color.matrix.flatMap { Self.lumaCoefficients[$0] }
            ?? (CVPixelBufferGetHeight(buffer) >= 720 ? (0.2126, 0.0722) : (0.299, 0.114))
        let kg = 1 - kr - kb
        let toRGB = simd_float4x4(columns: (
            SIMD4<Float>(1, 1, 1, 0),                                       // Y
            SIMD4<Float>(0, -2 * kb * (1 - kb) / kg, 2 * (1 - kb), 0),      // Cb
            SIMD4<Float>(2 * (1 - kr), -2 * kr * (1 - kr) / kg, 0, 0),      // Cr
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let transfer: Int32 = switch color.dynamicRangeName {
        case "HDR PQ": 1
        case "HDR HLG": 2
        default: 0
        }
        let bt2020: Int32 = color.primaries == (kCVImageBufferColorPrimaries_ITU_R_2020 as String) ? 1 : 0
        return YCbCrParameters(toRGB: toRGB, scale: scale, offset: offset, flags: SIMD4(transfer, bt2020, 0, 0))
    }
}
