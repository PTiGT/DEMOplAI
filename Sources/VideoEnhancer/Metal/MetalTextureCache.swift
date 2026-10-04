import CoreVideo
import Metal

/// Zero-copy Metal views of IOSurface-backed pixel buffers via `CVMetalTextureCache`.
///
/// The returned `CVMetalTexture` objects must stay alive until the GPU has finished with the
/// texture, so callers keep the `WrappedPlanes` value until their command buffer completes.
struct WrappedPlanes: @unchecked Sendable {
    let textures: [MTLTexture]
    fileprivate let cvTextures: [CVMetalTexture]
    let pixelBuffer: CVPixelBuffer
}

enum MetalTextureCache {
    enum WrapError: Error {
        case unsupportedFormat(OSType)
        case failed(CVReturn)
    }

    /// Plane formats for the pixel formats the pipeline accepts.
    static func planeFormats(for pixelFormat: OSType) -> [MTLPixelFormat]? {
        switch pixelFormat {
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
             kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
             kCVPixelFormatType_Lossless_420YpCbCr8BiPlanarVideoRange,
             kCVPixelFormatType_Lossless_420YpCbCr8BiPlanarFullRange:
            [.r8Unorm, .rg8Unorm]
        // The lossless 10-bit formats (&xv0/&xf0) are bit-packed without padding; reading them as
        // r16Unorm gives garbage (verified), so they are not accepted and trigger the 420v/x420 fallback.
        case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
             kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
            [.r16Unorm, .rg16Unorm]
        case kCVPixelFormatType_32BGRA:
            [.bgra8Unorm]
        case kCVPixelFormatType_64RGBAHalf:
            [.rgba16Float]
        case kCVPixelFormatType_OneComponent16Half:
            [.r16Float]
        default:
            nil
        }
    }

    static func wrap(_ buffer: CVPixelBuffer, cache: CVMetalTextureCache, usage: MTLTextureUsage = .shaderRead) throws -> WrappedPlanes {
        let pixelFormat = CVPixelBufferGetPixelFormatType(buffer)
        guard let formats = planeFormats(for: pixelFormat) else { throw WrapError.unsupportedFormat(pixelFormat) }
        let planar = CVPixelBufferIsPlanar(buffer)

        var cvTextures: [CVMetalTexture] = []
        var textures: [MTLTexture] = []
        for (plane, format) in formats.enumerated() {
            let width = planar ? CVPixelBufferGetWidthOfPlane(buffer, plane) : CVPixelBufferGetWidth(buffer)
            let height = planar ? CVPixelBufferGetHeightOfPlane(buffer, plane) : CVPixelBufferGetHeight(buffer)
            let attributes = [kCVMetalTextureUsage: usage.rawValue] as CFDictionary
            var cvTexture: CVMetalTexture?
            let status = CVMetalTextureCacheCreateTextureFromImage(
                nil, cache, buffer, attributes, format, width, height, planar ? plane : 0, &cvTexture
            )
            guard status == kCVReturnSuccess, let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else {
                throw WrapError.failed(status)
            }
            cvTextures.append(cvTexture)
            textures.append(texture)
        }
        return WrappedPlanes(textures: textures, cvTextures: cvTextures, pixelBuffer: buffer)
    }
}
