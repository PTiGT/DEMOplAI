import CoreMedia
import CoreVideo

/// One decoded frame on its way through the pipeline.
///
/// The pixel buffer is IOSurface-backed, so later stages wrap it as Metal textures
/// (`CVMetalTextureCache`) without copying pixels. Frames are treated as immutable after decode,
/// which is why passing them between threads is safe.
struct VideoFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    /// Media time of this frame in the player item's timeline.
    let presentationTime: CMTime
    /// Host time (CACurrentMediaTime base) of the vsync this frame was pulled for.
    let targetHostTime: CFTimeInterval
    /// Increments for every new frame pulled from the current item.
    let sequenceNumber: UInt64
    /// Clockwise display rotation (0/90/180/270) from the track transform; buffers are not rotated.
    let rotation: Int
    /// First frame after a seek or item change: temporal state (history, stabilization) must reset.
    let isDiscontinuity: Bool
    /// Nominal frame rate of the stream (0 if unknown).
    let nominalFrameRate: Double

    var width: Int { CVPixelBufferGetWidth(pixelBuffer) }
    var height: Int { CVPixelBufferGetHeight(pixelBuffer) }
    var pixelFormat: OSType { CVPixelBufferGetPixelFormatType(pixelBuffer) }
}

/// Facts about a pixel buffer for diagnostics, read from Core Video — nothing guessed.
struct PixelBufferDescription: Sendable, Equatable {
    var width: Int
    var height: Int
    var pixelFormat: OSType
    var planeCount: Int
    var isIOSurfaceBacked: Bool
    var color: ColorInfo

    init(_ buffer: CVPixelBuffer) {
        width = CVPixelBufferGetWidth(buffer)
        height = CVPixelBufferGetHeight(buffer)
        pixelFormat = CVPixelBufferGetPixelFormatType(buffer)
        planeCount = CVPixelBufferGetPlaneCount(buffer)
        isIOSurfaceBacked = CVPixelBufferGetIOSurface(buffer) != nil
        color = ColorInfo(pixelBuffer: buffer)
    }

    var formatSummary: String {
        "\(PixelFormat.fourCC(pixelFormat)) · \(PixelFormat.name(pixelFormat)) · \(planeCount) plane\(planeCount == 1 ? "" : "s")"
    }
}

enum PixelFormat {
    static func fourCC(_ code: OSType) -> String {
        let bytes = [24, 16, 8, 0].map { UInt8((code >> $0) & 0xFF) }
        return String(bytes: bytes, encoding: .macOSRoman) ?? "\(code)"
    }

    static func name(_ code: OSType) -> String {
        switch code {
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange: "NV12 8-bit, video range"
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange: "NV12 8-bit, full range"
        case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange: "4:2:0 10-bit, video range"
        case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange: "4:2:0 10-bit, full range"
        case kCVPixelFormatType_Lossless_420YpCbCr8BiPlanarVideoRange: "NV12 8-bit, lossless-compressed (decoder native)"
        case kCVPixelFormatType_Lossless_420YpCbCr8BiPlanarFullRange: "NV12 8-bit full range, lossless-compressed (decoder native)"
        case kCVPixelFormatType_Lossless_420YpCbCr10PackedBiPlanarVideoRange: "4:2:0 10-bit packed, lossless-compressed (decoder native)"
        case kCVPixelFormatType_32BGRA: "BGRA 8-bit"
        case kCVPixelFormatType_64RGBAHalf: "RGBA 16-bit float"
        default: "other"
        }
    }
}
