import Foundation
import Metal

/// A texture borrowed from `TexturePool`; it goes back to the pool when the last reference drops.
/// Whoever encodes GPU work with it keeps a reference until that command buffer completes.
final class PooledTexture: @unchecked Sendable {
    let texture: MTLTexture
    private weak var pool: TexturePool?

    fileprivate init(texture: MTLTexture, pool: TexturePool) {
        self.texture = texture
        self.pool = pool
    }

    var width: Int { texture.width }
    var height: Int { texture.height }

    deinit {
        pool?.recycle(texture)
    }
}

/// Reuses intermediate textures so steady-state playback allocates no GPU memory per frame.
final class TexturePool: @unchecked Sendable {
    private struct Key: Hashable {
        let width: Int
        let height: Int
        let format: MTLPixelFormat
        let usage: UInt
    }

    private let device: MTLDevice
    private let lock = NSLock()
    private var free: [Key: [MTLTexture]] = [:]
    private var allocatedBytes = 0
    /// Free textures kept per size/format; anything beyond is released.
    private let maxFreePerKey = 6

    init(device: MTLDevice) {
        self.device = device
    }

    func make(width: Int, height: Int, format: MTLPixelFormat = .rgba16Float,
              usage: MTLTextureUsage = [.shaderRead, .shaderWrite], label: String? = nil) -> PooledTexture {
        let key = Key(width: width, height: height, format: format, usage: usage.rawValue)
        let reused: MTLTexture? = lock.withLock { free[key]?.popLast() }
        if let reused {
            return PooledTexture(texture: reused, pool: self)
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false
        )
        descriptor.usage = usage
        descriptor.storageMode = .private
        let texture = device.makeTexture(descriptor: descriptor)!
        texture.label = label
        lock.withLock { allocatedBytes += texture.allocatedSize }
        return PooledTexture(texture: texture, pool: self)
    }

    fileprivate func recycle(_ texture: MTLTexture) {
        let key = Key(width: texture.width, height: texture.height, format: texture.pixelFormat, usage: texture.usage.rawValue)
        lock.withLock {
            if free[key, default: []].count < maxFreePerKey {
                free[key, default: []].append(texture)
            } else {
                allocatedBytes -= texture.allocatedSize
            }
        }
    }

    /// Drops all idle textures (e.g. after the output size changed).
    func purge() {
        lock.withLock {
            for textures in free.values {
                for texture in textures { allocatedBytes -= texture.allocatedSize }
            }
            free.removeAll()
        }
    }

    var bytesAllocated: Int { lock.withLock { allocatedBytes } }
}
