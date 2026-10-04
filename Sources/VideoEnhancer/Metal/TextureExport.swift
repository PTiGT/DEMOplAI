import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers

/// Writes an rgba16Float texture to an 8-bit PNG (debugging and pipeline verification).
enum TextureExport {
    static func writePNG(_ texture: MTLTexture, to url: URL, context: MetalContext) async throws {
        let width = texture.width, height = texture.height
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        descriptor.storageMode = .shared
        guard let staging = context.device.makeTexture(descriptor: descriptor),
              let commandBuffer = context.commandQueue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else { throw SRError.bufferAllocation(-3) }
        blit.copy(from: texture, to: staging)
        blit.endEncoding()
        commandBuffer.commit()
        await commandBuffer.completed()

        var halves = [Float16](repeating: 0, count: width * height * 4)
        staging.getBytes(&halves, bytesPerRow: width * 8, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            for c in 0..<3 {
                bytes[i * 4 + c] = UInt8(max(0, min(255, (Float(halves[i * 4 + c]) * 255).rounded())))
            }
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                            space: CGColorSpace(name: CGColorSpace.itur_709)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw SRError.bufferAllocation(-4)
        }
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
    }
}
