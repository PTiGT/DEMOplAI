// Verifies that Metal reads the decoder-native lossless-compressed format (&8v0) exactly like the
// uncompressed one (420v): grabs the same frame both ways, copies the luma/chroma planes to CPU
// through Metal blits and compares bytes.
// Usage: swift Tools/compare_native_format.swift <video> [seconds]
import AVFoundation
import CoreVideo
import Metal

let url = URL(fileURLWithPath: CommandLine.arguments[1])
let time = CMTime(seconds: CommandLine.arguments.count > 2 ? Double(CommandLine.arguments[2])! : 1.0, preferredTimescale: 600)
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
var cache: CVMetalTextureCache?
CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)

func grab(format: OSType?) -> CVPixelBuffer {
    var attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:], kCVPixelBufferMetalCompatibilityKey as String: true]
    if let format { attrs[kCVPixelBufferPixelFormatTypeKey as String] = format }
    let item = AVPlayerItem(url: url)
    let output = AVPlayerItemVideoOutput(pixelBufferAttributes: attrs)
    item.add(output)
    let player = AVPlayer(playerItem: item)
    var done = false
    player.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero) { _ in done = true }
    while !done { RunLoop.main.run(until: Date().addingTimeInterval(0.01)) }
    for _ in 0..<200 {
        if let pb = output.copyPixelBuffer(forItemTime: time, itemTimeForDisplay: nil) { return pb }
        RunLoop.main.run(until: Date().addingTimeInterval(0.01))
    }
    fatalError("no frame")
}

// Reads the plane through a compute shader (what the app's converter sees) into a byte buffer.
let library = try! device.makeLibrary(source: """
#include <metal_stdlib>
using namespace metal;
kernel void readPlane(texture2d<float, access::read> t [[texture(0)]], device uchar2 *out [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= t.get_width() || gid.y >= t.get_height()) return;
    float4 v = t.read(gid);
    out[gid.y * t.get_width() + gid.x] = uchar2(round(v.rg * 255.0));
}
""", options: nil)
let readPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "readPlane")!)

func planeBytes(_ pb: CVPixelBuffer, plane: Int, format: MTLPixelFormat, bpp: Int) -> [UInt8] {
    let w = CVPixelBufferGetWidthOfPlane(pb, plane), h = CVPixelBufferGetHeightOfPlane(pb, plane)
    var cvTex: CVMetalTexture?
    CVMetalTextureCacheCreateTextureFromImage(nil, cache!, pb, nil, format, w, h, plane, &cvTex)
    let tex = CVMetalTextureGetTexture(cvTex!)!
    let buffer = device.makeBuffer(length: w * h * 2, options: .storageModeShared)!
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(readPipeline)
    enc.setTexture(tex, index: 0)
    enc.setBuffer(buffer, offset: 0, index: 0)
    enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let all = Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt8.self), count: w * h * 2))
    // Keep only the channels the format has (r for luma, rg for chroma).
    return bpp == 1 ? stride(from: 0, to: all.count, by: 2).map { all[$0] } : all
}

let plain = grab(format: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
let native = grab(format: nil)
func fourcc(_ c: OSType) -> String { String(bytes: [24,16,8,0].map { UInt8((c >> $0) & 0xFF) }, encoding: .macOSRoman)! }
print("plain: \(fourcc(CVPixelBufferGetPixelFormatType(plain)))  native: \(fourcc(CVPixelBufferGetPixelFormatType(native)))")
for (plane, format, bpp, name) in [(0, MTLPixelFormat.r8Unorm, 1, "luma"), (1, .rg8Unorm, 2, "chroma")] {
    let a = planeBytes(plain, plane: plane, format: format, bpp: bpp)
    let b = planeBytes(native, plane: plane, format: format, bpp: bpp)
    var maxDiff = 0, differing = 0
    for i in 0..<a.count { let d = abs(Int(a[i]) - Int(b[i])); if d > 0 { differing += 1; maxDiff = max(maxDiff, d) } }
    print("\(name): \(a.count) bytes, differing \(differing), max diff \(maxDiff)")
}
