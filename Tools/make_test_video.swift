// Generates a synthetic H.264 test clip with fine detail and motion, for checking playback
// and (later) comparing upscalers and measuring flicker. Encoding uses VideoToolbox via AVAssetWriter.
//
// Usage: swift Tools/make_test_video.swift <output.mp4> [width] [height] [fps] [seconds] [h264|hevc10] [photo]
//   With a photo, frames are a slow pan/zoom over it (natural texture, realistic for SR tests).
// Example: swift Tools/make_test_video.swift TestMedia/pattern_720p60.mp4 1280 720 60 10
//          swift Tools/make_test_video.swift TestMedia/pattern_1080p24_hevc10.mp4 1920 1080 24 6 hevc10
//          swift Tools/make_test_video.swift TestMedia/natural_360p24.mp4 640 360 24 20 h264 TestMedia/images/aerial_4k.png

import AVFoundation
import CoreGraphics
import CoreText
import Foundation
import ImageIO
import VideoToolbox

let args = CommandLine.arguments
guard args.count >= 2 else {
    print("Usage: make_test_video.swift <output.mp4> [width] [height] [fps] [seconds] [h264|hevc10]")
    exit(1)
}
let outputURL = URL(fileURLWithPath: args[1])
let width = args.count > 2 ? Int(args[2])! : 1280
let height = args.count > 3 ? Int(args[3])! : 720
let fps = args.count > 4 ? Int32(args[4])! : 60
let seconds = args.count > 5 ? Int(args[5])! : 10
let hevc10 = args.count > 6 && args[6] == "hevc10"
let photo: CGImage? = args.count > 7 ? {
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: args[7]) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(source, 0, nil)
}() : nil
let frameCount = Int(fps) * seconds

try? FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
try? FileManager.default.removeItem(at: outputURL)

let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
var compression: [String: Any] = [
    // Photo mode uses a streaming-like bitrate so codec artifacts look like real downloaded video.
    AVVideoAverageBitRateKey: photo == nil ? width * height * 4 : width * height * Int(fps) / 12,
    AVVideoExpectedSourceFrameRateKey: fps,
    AVVideoMaxKeyFrameIntervalKey: Int(fps) * 2,
]
if hevc10 {
    compression[AVVideoProfileLevelKey] = kVTProfileLevel_HEVC_Main10_AutoLevel as String
}
let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
    AVVideoCodecKey: hevc10 ? AVVideoCodecType.hevc : AVVideoCodecType.h264,
    AVVideoWidthKey: width,
    AVVideoHeightKey: height,
    AVVideoCompressionPropertiesKey: compression,
    AVVideoColorPropertiesKey: [
        AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
        AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
        AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
    ],
])
input.expectsMediaDataInRealTime = false
// HEVC Main10 is fed 10-bit 4:2:0 buffers; the BGRA drawing is converted with VTPixelTransferSession.
let writerPixelFormat = hevc10 ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange : kCVPixelFormatType_32BGRA
let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
    kCVPixelBufferPixelFormatTypeKey as String: writerPixelFormat,
    kCVPixelBufferWidthKey as String: width,
    kCVPixelBufferHeightKey as String: height,
])
writer.add(input)
writer.startWriting()
writer.startSession(atSourceTime: .zero)

let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
let font = CTFontCreateWithName("Menlo-Bold" as CFString, CGFloat(height) / 14, nil)
let smallFont = CTFontCreateWithName("Helvetica" as CFString, CGFloat(height) / 48, nil)

func drawText(_ text: String, font: CTFont, at point: CGPoint, in ctx: CGContext) {
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 1, alpha: 1),
    ]
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attributes))
    ctx.textPosition = point
    CTLineDraw(line, ctx)
}

/// Pan/zoom over the photo: the crop window drifts and slowly zooms in (Ken Burns).
func renderPhoto(_ image: CGImage, frame: Int, into ctx: CGContext) {
    let t = Double(frame) / Double(fps) / Double(seconds)
    let iw = Double(image.width), ih = Double(image.height)
    let zoom = 1.0 + 0.35 * t
    let cropW = iw * 0.55 / zoom, cropH = cropW * Double(height) / Double(width)
    let cx = iw * (0.35 + 0.3 * t), cy = ih * (0.55 - 0.1 * t)
    let crop = CGRect(x: cx - cropW / 2, y: cy - cropH / 2, width: cropW, height: cropH)
    // Draw the whole image transformed so that `crop` fills the frame (sub-pixel motion preserved).
    let sx = Double(width) / cropW, sy = Double(height) / cropH
    ctx.interpolationQuality = .high
    ctx.draw(image, in: CGRect(x: -crop.minX * sx, y: -(ih - crop.maxY) * sy, width: iw * sx, height: ih * sy))
}

func render(frame: Int, into buffer: CVPixelBuffer) {
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let ctx = CGContext(
        data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    )!
    if let photo {
        renderPhoto(photo, frame: frame, into: ctx)
        return
    }
    let t = Double(frame) / Double(fps)
    let w = CGFloat(width), h = CGFloat(height)

    // Slowly shifting background gradient.
    let hue = CGFloat((sin(t * 0.4) + 1) / 2)
    let gradient = CGGradient(colorsSpace: colorSpace, colors: [
        CGColor(red: 0.1 + 0.3 * hue, green: 0.15, blue: 0.35, alpha: 1),
        CGColor(red: 0.05, green: 0.25 + 0.2 * hue, blue: 0.2, alpha: 1),
    ] as CFArray, locations: [0, 1])!
    ctx.drawLinearGradient(gradient, start: .zero, end: CGPoint(x: w, y: h), options: [])

    // Static zone plate (left): fine radial frequencies reveal aliasing and SR hallucination.
    let plate = CGRect(x: w * 0.04, y: h * 0.2, width: h * 0.55, height: h * 0.55)
    let rings = 48
    for i in stride(from: rings, to: 0, by: -1) {
        let r = plate.width / 2 * sqrt(CGFloat(i) / CGFloat(rings))
        ctx.setFillColor(gray: i % 2 == 0 ? 0.95 : 0.05, alpha: 1)
        ctx.fillEllipse(in: CGRect(x: plate.midX - r, y: plate.midY - r, width: 2 * r, height: 2 * r))
    }

    // Moving fine stripes (center): sub-pixel motion makes temporal flicker visible.
    let stripesRect = CGRect(x: w * 0.42, y: h * 0.2, width: w * 0.22, height: h * 0.55)
    ctx.saveGState()
    ctx.clip(to: stripesRect)
    ctx.setFillColor(gray: 0.9, alpha: 1)
    let offset = CGFloat(t * 13.7).truncatingRemainder(dividingBy: 6)
    var x = stripesRect.minX - 6 + offset
    while x < stripesRect.maxX {
        ctx.fill(CGRect(x: x, y: stripesRect.minY, width: 1.5, height: stripesRect.height))
        x += 3 + (x - stripesRect.minX) / stripesRect.width * 3
    }
    ctx.restoreGState()

    // Orbiting disc with a checkerboard (right): large motion.
    let cx = w * 0.82 + cos(t * 1.3) * w * 0.08
    let cy = h * 0.5 + sin(t * 1.3) * h * 0.2
    let cell: CGFloat = 6
    for row in 0..<16 {
        for col in 0..<16 {
            ctx.setFillColor(gray: (row + col) % 2 == 0 ? 1 : 0.1, alpha: 1)
            ctx.fill(CGRect(x: cx - 8 * cell + CGFloat(col) * cell, y: cy - 8 * cell + CGFloat(row) * cell, width: cell, height: cell))
        }
    }

    drawText(String(format: "Frame %05d  %.3f s", frame, t), font: font, at: CGPoint(x: w * 0.04, y: h * 0.82), in: ctx)
    drawText("Small text for detail check: The quick brown fox jumps over the lazy dog 0123456789",
             font: smallFont, at: CGPoint(x: w * 0.04, y: h * 0.08), in: ctx)
}

var transferSession: VTPixelTransferSession?
var drawingBuffer: CVPixelBuffer?
if hevc10 {
    VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &transferSession)
    CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                        [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary, &drawingBuffer)
}

for frame in 0..<frameCount {
    while !input.isReadyForMoreMediaData { usleep(1000) }
    var buffer: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &buffer)
    if let transferSession, let drawingBuffer {
        render(frame: frame, into: drawingBuffer)
        CVBufferSetAttachment(buffer!, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(buffer!, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(buffer!, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
        VTPixelTransferSessionTransferImage(transferSession, from: drawingBuffer, to: buffer!)
    } else {
        render(frame: frame, into: buffer!)
    }
    adaptor.append(buffer!, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: fps))
}
input.markAsFinished()

let done = DispatchSemaphore(value: 0)
writer.finishWriting { done.signal() }
done.wait()
if writer.status == .completed {
    print("Wrote \(outputURL.path): \(width)×\(height) @ \(fps) fps, \(seconds) s, \(hevc10 ? "HEVC Main10" : "H.264")")
} else {
    print("Failed: \(writer.error?.localizedDescription ?? "unknown error")")
    exit(1)
}
