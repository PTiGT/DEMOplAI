import Foundation

/// Wires the playback side (AVPlayer, frame output) to the processing and display side
/// (engine, Metal renderer, enhancement settings). Owned by the app delegate for the app's lifetime.
@MainActor
final class VideoPlayer {
    let playback: PlaybackController
    let monitor = PerformanceMonitor()
    let engine: VideoProcessingEngine?
    let renderer: MetalRenderer?
    let enhancement: EnhancementController
    /// Why the Metal pipeline could not start, if it could not.
    let setupError: String?

    init() {
        playback = PlaybackController()
        switch MetalContext.shared {
        case .success(let context):
            let engine = VideoProcessingEngine(context: context, frameProvider: playback.frameProvider, monitor: monitor)
            self.engine = engine
            renderer = MetalRenderer(context: context, engine: engine)
            setupError = nil
        case .failure(let error):
            engine = nil
            renderer = nil
            setupError = "Metal pipeline unavailable: \(error.localizedDescription)"
        }
        enhancement = EnhancementController(engine: engine, renderer: renderer)
        playback.onMediaClosed = { [weak self] in self?.renderer?.clear() }
        playback.onVideoInfoChange = { [weak self] info in
            self?.monitor.reset()
            self?.engine?.resetDynamicQuality()
            self?.enhancement.videoDidChange(info)
        }
    }
}
