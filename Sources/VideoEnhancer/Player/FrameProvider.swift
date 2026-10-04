import AVFoundation
import CoreVideo
import Foundation

/// Pulls decoded frames out of the playing `AVPlayerItem` as `CVPixelBuffer`s.
///
/// AVPlayer keeps owning decoding (VideoToolbox), audio and A/V sync; this class only attaches an
/// `AVPlayerItemVideoOutput` and, on every vsync, asks it for the frame that should be on screen at
/// that vsync's host time. Nothing is queued here: if a newer frame exists it replaces the older one,
/// so a slow consumer can never build up latency (the start of the backpressure chain).
///
/// Thread-safe: `attach`/`detach` run on the main thread, `copyFrame` may run on a render thread.
final class FrameProvider: @unchecked Sendable {
    enum PixelFormatPolicy: Hashable, Sendable {
        /// Decoder native for 8-bit H.264/HEVC, `matchSource` otherwise.
        case automatic
        /// Bi-planar 4:2:0 at the source bit depth (`420v`/`420f`/`x420`/`xf20`).
        case matchSource
        /// No format requested: whatever the decoder produces. On Apple Silicon this is a
        /// lossless-compressed format (`&8v0` for 8-bit): no conversion pass, CPU cannot read it.
        /// Metal reads `&8v0` bit-identically to `420v` (Tools/compare_native_format.swift).
        case decoderNative

        func resolved(for info: VideoInfo) -> PixelFormatPolicy {
            guard self == .automatic else { return self }
            let nativeReadable = info.bitDepth == 8 && !info.color.isHDR && ["H.264", "HEVC"].contains(info.codecName)
            return nativeReadable ? .decoderNative : .matchSource
        }
    }

    struct Statistics: Sendable {
        var isAttached = false
        var requestedFormat: String = "—"
        var lastFrame: PixelBufferDescription?
        var lastPresentationTime: Double?
        var framesReceived: UInt64 = 0
        /// New frames per second of wall time, measured over the last second.
        var measuredFPS: Double = 0
        /// Source frames that were never pulled because a newer one was already due (PTS gaps).
        var framesNotPulled: UInt64 = 0
        var vsyncRate: Double = 0
        var unusableFrameError: String?
    }

    private let lock = NSLock()
    private var output: AVPlayerItemVideoOutput?
    private weak var attachedItem: AVPlayerItem?
    private var nominalFrameDuration: Double?
    private var rotation = 0
    private var nominalFrameRate = 0.0
    private var discontinuityPending = true
    private var unusableFrameError: String?
    private var stats = Statistics()
    private var recentFrameHostTimes: [CFTimeInterval] = []
    private var previousPTS: Double?
    private var previousVsyncHostTime: CFTimeInterval?

    /// Attaches a fresh video output to `item`, replacing any output on a previous item.
    func attach(to item: AVPlayerItem, info: VideoInfo, policy: PixelFormatPolicy = .automatic) {
        var attributes: [String: any Sendable] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: any Sendable](),
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        let requested: String
        switch policy.resolved(for: info) {
        case .decoderNative:
            requested = policy == .automatic ? "decoder native (automatic)" : "decoder native"
        default:
            attributes[kCVPixelBufferPixelFormatTypeKey as String] = info.matchingPixelFormat
            requested = PixelFormat.fourCC(info.matchingPixelFormat) + (policy == .automatic ? " (automatic)" : "")
        }

        let newOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: attributes)
        detach()
        item.add(newOutput)

        lock.withLock {
            output = newOutput
            attachedItem = item
            nominalFrameDuration = info.nominalFrameDuration
            rotation = info.rotation
            nominalFrameRate = info.nominalFrameRate
            discontinuityPending = true
            unusableFrameError = nil
            stats = Statistics(isAttached: true, requestedFormat: requested)
            recentFrameHostTimes.removeAll(keepingCapacity: true)
            previousPTS = nil
        }
    }

    func detach() {
        let (oldOutput, oldItem) = lock.withLock {
            defer {
                output = nil
                attachedItem = nil
                stats = Statistics()
            }
            return (output, attachedItem)
        }
        if let oldOutput, let oldItem {
            oldItem.remove(oldOutput)
        }
    }

    /// Call after a seek so the jump in presentation time is not counted as skipped frames.
    func markDiscontinuity() {
        lock.withLock {
            previousPTS = nil
            discontinuityPending = true
        }
    }

    /// Playback rate measured from the item clock (0 when paused, 2 at 2× speed).
    func playbackRate(atHostTime hostTime: CFTimeInterval) -> Double {
        lock.withLock {
            guard let output else { return 0 }
            let a = output.itemTime(forHostTime: hostTime)
            let b = output.itemTime(forHostTime: hostTime + 0.1)
            guard a.isNumeric, b.isNumeric else { return 0 }
            return (b.seconds - a.seconds) / 0.1
        }
    }

    /// Returns the frame due at `hostTime` if it differs from the last returned one, else nil.
    func copyFrame(forHostTime hostTime: CFTimeInterval) -> VideoFrame? {
        lock.withLock {
            if let previousVsyncHostTime, hostTime > previousVsyncHostTime {
                let rate = 1 / (hostTime - previousVsyncHostTime)
                stats.vsyncRate = stats.vsyncRate == 0 ? rate : stats.vsyncRate * 0.95 + rate * 0.05
            }
            previousVsyncHostTime = hostTime

            guard let output else { return nil }
            let itemTime = output.itemTime(forHostTime: hostTime)
            guard output.hasNewPixelBuffer(forItemTime: itemTime) else { return nil }

            var displayTime = CMTime.invalid
            guard let buffer = output.copyPixelBuffer(forItemTime: itemTime, itemTimeForDisplay: &displayTime) else {
                return nil
            }

            let pts = displayTime.isNumeric ? displayTime : itemTime
            record(ptsSeconds: pts.seconds, hostTime: hostTime, buffer: buffer)
            let discontinuity = discontinuityPending
            discontinuityPending = false
            return VideoFrame(
                pixelBuffer: buffer,
                presentationTime: pts,
                targetHostTime: hostTime,
                sequenceNumber: stats.framesReceived,
                rotation: rotation,
                isDiscontinuity: discontinuity,
                nominalFrameRate: nominalFrameRate
            )
        }
    }

    /// Called (once per item, on the main queue) when delivered frames cannot be used by Metal,
    /// so the owner can re-attach with an explicit format.
    var onUnusableFrames: (@Sendable () -> Void)?

    /// The pipeline could not use a delivered frame (e.g. unsupported pixel format for Metal).
    func reportUnusableFrame(_ error: Error) {
        let firstReport = lock.withLock {
            defer { unusableFrameError = "\(error)" }
            return unusableFrameError == nil
        }
        if firstReport, let onUnusableFrames {
            DispatchQueue.main.async { onUnusableFrames() }
        }
    }

    func statistics() -> Statistics {
        lock.withLock {
            var snapshot = stats
            snapshot.unusableFrameError = unusableFrameError
            let now = CACurrentMediaTime()
            // A paused player delivers no frames: report 0 instead of the last value.
            snapshot.measuredFPS = recentFrameHostTimes.filter { now - $0 <= 1.0 }.count > 1 ? stats.measuredFPS : 0
            return snapshot
        }
    }

    // Called with the lock held.
    private func record(ptsSeconds: Double, hostTime: CFTimeInterval, buffer: CVPixelBuffer) {
        stats.framesReceived += 1
        stats.lastPresentationTime = ptsSeconds
        // Re-reading attachments for every frame is cheap, but the description only changes on format switches.
        if stats.lastFrame == nil || stats.framesReceived % 30 == 0 {
            stats.lastFrame = PixelBufferDescription(buffer)
        }

        if let previousPTS, let frameDuration = nominalFrameDuration {
            let gap = ptsSeconds - previousPTS
            // Negative or very large gaps are seeks / loops, not skipped frames.
            if gap > frameDuration * 1.5, gap < 1.0 {
                stats.framesNotPulled += UInt64((gap / frameDuration).rounded()) - 1
            }
        }
        previousPTS = ptsSeconds

        recentFrameHostTimes.append(hostTime)
        recentFrameHostTimes.removeAll { hostTime - $0 > 1.0 }
        if let first = recentFrameHostTimes.first, recentFrameHostTimes.count > 1, hostTime > first {
            stats.measuredFPS = Double(recentFrameHostTimes.count - 1) / (hostTime - first)
        }
    }
}
