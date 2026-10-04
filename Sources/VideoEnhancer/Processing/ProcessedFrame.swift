import CoreMedia
import Foundation

/// How the enhanced image of a frame was produced. Shown in the UI and the performance monitor,
/// so it must always describe what really happened to that frame.
enum EnhancementPath: Sendable, Equatable {
    case original
    case metal
    case ai(model: String)

    var label: String {
        switch self {
        case .original: "Original"
        case .metal: "Metal Enhancement"
        case .ai(let model): "AI · \(model)"
        }
    }
}

/// One video frame ready for display: the color-converted source plus, optionally, its enhanced version.
/// Holding this object keeps both textures out of the pool.
final class ProcessedFrame: @unchecked Sendable {
    let sequence: UInt64
    let presentationTime: CMTime
    /// Vsync host time at which the frame was pulled from the decoder.
    let pulledAt: CFTimeInterval
    /// Host time at which the frame should be on screen (pull time + lookahead).
    let dueAt: CFTimeInterval
    let original: PooledTexture
    let enhanced: PooledTexture?
    let path: EnhancementPath
    /// Clockwise display rotation in degrees (0, 90, 180, 270) from the track transform.
    let rotation: Int
    /// Engine epoch (incremented on seek / item change); frames from an older epoch are discarded.
    let epoch: Int

    init(sequence: UInt64, presentationTime: CMTime, pulledAt: CFTimeInterval, dueAt: CFTimeInterval,
         original: PooledTexture, enhanced: PooledTexture?, path: EnhancementPath, rotation: Int, epoch: Int = 0) {
        self.sequence = sequence
        self.presentationTime = presentationTime
        self.pulledAt = pulledAt
        self.dueAt = dueAt
        self.original = original
        self.enhanced = enhanced
        self.path = path
        self.rotation = rotation
        self.epoch = epoch
    }

    /// Size of the picture as displayed (after rotation).
    var displaySize: CGSize {
        let w = CGFloat(original.width), h = CGFloat(original.height)
        return rotation % 180 == 0 ? CGSize(width: w, height: h) : CGSize(width: h, height: w)
    }
}

enum AspectFit {
    /// Largest rect with the aspect ratio of `content` centered in `bounds` (pixel-aligned).
    static func rect(content: CGSize, in bounds: CGSize) -> CGRect {
        guard content.width > 0, content.height > 0, bounds.width > 0, bounds.height > 0 else { return .zero }
        let scale = min(bounds.width / content.width, bounds.height / content.height)
        let size = CGSize(width: (content.width * scale).rounded(), height: (content.height * scale).rounded())
        return CGRect(x: ((bounds.width - size.width) / 2).rounded(), y: ((bounds.height - size.height) / 2).rounded(),
                      width: size.width, height: size.height)
    }
}
