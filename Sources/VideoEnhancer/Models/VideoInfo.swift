import AVFoundation
import VideoToolbox

/// Facts about the opened file, read from the container — nothing here is estimated.
struct VideoInfo: Sendable, Equatable {
    /// Display size, i.e. after applying the track's rotation transform.
    var width: Int
    var height: Int
    var nominalFrameRate: Double
    var codecFourCC: String
    var codecName: String
    /// `VTIsHardwareDecodeSupported` for this codec on this Mac.
    var hardwareDecodeSupported: Bool
    /// From the format description; 8 when the container does not say.
    var bitDepth: Int
    var isFullRange: Bool
    var color: ColorInfo
    var duration: Double
    var hasAudio: Bool
    /// Clockwise rotation (0/90/180/270) the track transform applies for display.
    var rotation: Int

    var resolutionText: String { "\(width)×\(height)" }

    var summary: String {
        var parts = [resolutionText]
        if nominalFrameRate > 0 {
            parts.append("\(nominalFrameRate.formatted(.number.precision(.fractionLength(0...2)))) fps")
        }
        parts.append(codecName + (hardwareDecodeSupported ? " (HW decode)" : " (SW decode)"))
        parts.append("\(bitDepth)-bit \(color.dynamicRangeName)")
        return parts.joined(separator: " · ")
    }

    /// Bi-planar 4:2:0 output matching the source, so AVFoundation does not reduce bit depth
    /// and Metal reads the planes directly (Stage 3).
    var matchingPixelFormat: OSType {
        switch (bitDepth > 8 || color.isHDR, isFullRange) {
        case (false, false): kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        case (false, true): kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        case (true, false): kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        case (true, true): kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
        }
    }

    /// Duration of one frame at the nominal rate, or nil for variable/unknown frame rate.
    var nominalFrameDuration: Double? {
        nominalFrameRate > 0 ? 1 / nominalFrameRate : nil
    }

    enum LoadError: LocalizedError {
        case notPlayable
        case noVideoTrack

        var errorDescription: String? {
            switch self {
            case .notPlayable: "AVFoundation cannot play this file format."
            case .noVideoTrack: "The file has no video track."
            }
        }
    }

    static func load(from asset: AVURLAsset) async throws -> VideoInfo {
        guard try await asset.load(.isPlayable) else { throw LoadError.notPlayable }
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw LoadError.noVideoTrack
        }

        let duration = try await asset.load(.duration)
        let (naturalSize, transform, frameRate, formats) = try await track.load(
            .naturalSize, .preferredTransform, .nominalFrameRate, .formatDescriptions
        )
        let hasAudio = try await !asset.loadTracks(withMediaType: .audio).isEmpty

        let displayRect = CGRect(origin: .zero, size: naturalSize).applying(transform)
        let format = formats.first
        let codecType = format.map(CMFormatDescriptionGetMediaSubType) ?? 0
        let bitDepth = format.flatMap {
            CMFormatDescriptionGetExtension($0, extensionKey: kCMFormatDescriptionExtension_BitsPerComponent) as? Int
        } ?? 8
        let isFullRange = format.flatMap {
            CMFormatDescriptionGetExtension($0, extensionKey: kCMFormatDescriptionExtension_FullRangeVideo) as? Bool
        } ?? false

        return VideoInfo(
            width: Int(abs(displayRect.width).rounded()),
            height: Int(abs(displayRect.height).rounded()),
            nominalFrameRate: Double(frameRate),
            codecFourCC: fourCCString(codecType),
            codecName: codecName(for: codecType),
            hardwareDecodeSupported: codecType != 0 && VTIsHardwareDecodeSupported(codecType),
            bitDepth: bitDepth,
            isFullRange: isFullRange,
            color: format.map(ColorInfo.init(formatDescription:)) ?? ColorInfo(matrix: nil, primaries: nil, transferFunction: nil),
            duration: duration.isNumeric ? duration.seconds : 0,
            hasAudio: hasAudio,
            rotation: rotationDegrees(transform)
        )
    }

    private static func rotationDegrees(_ t: CGAffineTransform) -> Int {
        let degrees = Int((atan2(t.b, t.a) * 180 / .pi).rounded())
        return ((degrees % 360) + 360) % 360 / 90 * 90
    }

    private static func codecName(for type: FourCharCode) -> String {
        switch fourCCString(type) {
        case "avc1", "avc3": "H.264"
        case "hvc1", "hev1": "HEVC"
        case "dvh1", "dvhe": "HEVC (Dolby Vision)"
        case "av01": "AV1"
        case "vp09": "VP9"
        case "apco": "ProRes 422 Proxy"
        case "apcs": "ProRes 422 LT"
        case "apcn": "ProRes 422"
        case "apch": "ProRes 422 HQ"
        case "ap4h": "ProRes 4444"
        case "ap4x": "ProRes 4444 XQ"
        case "mp4v": "MPEG-4 Part 2"
        case "jpeg": "Motion JPEG"
        case let other: other.trimmingCharacters(in: .whitespaces)
        }
    }

    private static func fourCCString(_ code: FourCharCode) -> String {
        let bytes = [24, 16, 8, 0].map { UInt8((code >> $0) & 0xFF) }
        return String(bytes: bytes, encoding: .macOSRoman) ?? "\(code)"
    }
}
