import AppKit
import AVFoundation
import Observation
import UniformTypeIdentifiers

/// Owns the `AVPlayer` and exposes playback state to the UI.
///
/// AVPlayer decodes (VideoToolbox under the hood) and owns timing, audio and A/V sync.
/// `frameProvider` taps the decoded frames as `CVPixelBuffer`s for the processing pipeline;
/// until the Metal renderer exists (Stage 3) `AVPlayerLayer` still does the on-screen display.
///
/// Lives for the whole app lifetime (owned by `AppDelegate`), so observers are never torn down.
@MainActor
@Observable
final class PlaybackController {
    static let availableRates: [Float] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    let player = AVPlayer()
    let frameProvider = FrameProvider()

    private(set) var playlist = Playlist()
    private(set) var videoInfo: VideoInfo?
    private(set) var isPlaying = false
    private(set) var currentTime: Double = 0
    private(set) var duration: Double = 0
    private(set) var volume: Float
    private(set) var isMuted = false
    private(set) var rate: Float = 1.0
    private(set) var pixelFormatPolicy: FrameProvider.PixelFormatPolicy = .automatic
    var errorMessage: String?

    var hasMedia: Bool { playlist.current != nil }
    /// Called when the current media goes away (e.g. it failed to open) so the display can clear.
    @ObservationIgnored var onMediaClosed: (() -> Void)?
    /// Called on the main thread when a new item's track info is known.
    @ObservationIgnored var onVideoInfoChange: ((VideoInfo?) -> Void)?
    var title: String { playlist.current?.lastPathComponent ?? "Video Enhancer" }

    @ObservationIgnored private var timeObserver: Any?
    @ObservationIgnored private var playerObservations: [NSKeyValueObservation] = []
    @ObservationIgnored private var itemObservations: [NSKeyValueObservation] = []
    @ObservationIgnored private var endObserver: NSObjectProtocol?
    @ObservationIgnored private var infoTask: Task<Void, Never>?

    @ObservationIgnored private var isScrubbing = false
    @ObservationIgnored private var wasPlayingBeforeScrub = false
    @ObservationIgnored private var seekInFlight = false
    @ObservationIgnored private var pendingSeek: (time: CMTime, precise: Bool)?

    private enum Keys {
        static let volume = "playback.volume"
    }

    init() {
        volume = UserDefaults.standard.object(forKey: Keys.volume) as? Float ?? 1.0
        player.volume = volume
        player.actionAtItemEnd = .pause
        player.preventsDisplaySleepDuringVideoPlayback = true
        installPlayerObservers()
        frameProvider.onUnusableFrames = { [weak self] in
            MainActor.assumeIsolated { self?.fallBackToExplicitFrameFormat() }
        }
    }

    // MARK: - Opening media

    func presentOpenPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.message = "Choose one video (its folder becomes the playlist) or several videos."
        guard panel.runModal() == .OK else { return }
        open(panel.urls)
    }

    /// One file: its folder becomes the playlist (for Previous / Next). Several files: exactly those.
    func open(_ urls: [URL]) {
        let files = urls.filter(\.isFileURL)
        let videos = files.filter(Playlist.isVideoFile)
        guard !videos.isEmpty else {
            if let first = files.first {
                errorMessage = "\(first.lastPathComponent) is not a video file."
            }
            return
        }
        playlist = videos.count == 1 ? Playlist.folder(containing: videos[0]) : Playlist(items: videos)
        loadCurrentItem(autoplay: true)
    }

    func next() {
        guard playlist.moveToNext() != nil else { return }
        loadCurrentItem(autoplay: true)
    }

    /// Restarts the current file if more than 3 s have played, otherwise goes to the previous file.
    func previous() {
        guard hasMedia else { return }
        if currentTime > 3 || !playlist.hasPrevious {
            seek(to: 0, precise: true)
            return
        }
        playlist.moveToPrevious()
        loadCurrentItem(autoplay: true)
    }

    private func loadCurrentItem(autoplay: Bool) {
        guard let url = playlist.current else { return }

        tearDownItemObservers()
        frameProvider.detach()
        errorMessage = nil
        videoInfo = nil
        currentTime = 0
        duration = 0
        pendingSeek = nil

        // The frame output's pixel format depends on the track (bit depth, range), so the track is
        // inspected first and the output is attached before the item reaches the player. Attaching
        // to an already prepared item makes AVFoundation rebuild its render pipeline, which snaps
        // the playhead to a keyframe.
        let asset = AVURLAsset(url: url)
        infoTask?.cancel()
        infoTask = Task { [weak self] in
            do {
                let info = try await VideoInfo.load(from: asset)
                guard !Task.isCancelled, let self else { return }
                self.videoInfo = info
                self.duration = info.duration
                self.onVideoInfoChange?(info)

                let item = AVPlayerItem(asset: asset)
                self.frameProvider.attach(to: item, info: info, policy: self.pixelFormatPolicy)
                self.observe(item)
                self.player.replaceCurrentItem(with: item)
                if autoplay {
                    self.player.play()
                }
            } catch {
                guard !Task.isCancelled, let self else { return }
                self.player.replaceCurrentItem(with: nil)
                self.onMediaClosed?()
                self.errorMessage = "Cannot open \(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
    }

    // MARK: - Transport

    func togglePlayPause() {
        guard hasMedia else { return }
        if player.timeControlStatus == .paused {
            if duration > 0, currentTime >= duration - 0.1 {
                seek(to: 0, precise: true)
            }
            player.play()
        } else {
            player.pause()
        }
    }

    func step(by seconds: Double) {
        guard hasMedia else { return }
        seek(to: currentTime + seconds, precise: true)
    }

    func beginScrubbing() {
        isScrubbing = true
        wasPlayingBeforeScrub = player.timeControlStatus != .paused
        player.pause()
    }

    /// Called continuously while the seek bar is dragged; uses fast (keyframe-tolerant) seeks.
    func scrub(to seconds: Double) {
        seek(to: seconds, precise: !isScrubbing)
    }

    func endScrubbing() {
        isScrubbing = false
        seek(to: currentTime, precise: true)
        if wasPlayingBeforeScrub {
            player.play()
        }
    }

    /// Coalesces seeks: while one is in flight only the latest requested target is kept,
    /// so dragging the seek bar never queues up a backlog of stale seeks.
    func seek(to seconds: Double, precise: Bool) {
        let upperBound = duration > 0 ? duration : seconds
        let clamped = min(max(seconds, 0), upperBound)
        currentTime = clamped
        frameProvider.markDiscontinuity()
        pendingSeek = (CMTime(seconds: clamped, preferredTimescale: 600), precise)
        if !seekInFlight {
            performPendingSeek()
        }
    }

    private func performPendingSeek() {
        guard let request = pendingSeek else { return }
        pendingSeek = nil
        seekInFlight = true
        let tolerance = request.precise ? CMTime.zero : CMTime(seconds: 0.5, preferredTimescale: 600)
        player.seek(to: request.time, toleranceBefore: tolerance, toleranceAfter: tolerance) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.seekInFlight = false
                self.performPendingSeek()
            }
        }
    }

    /// Re-attaches the frame output of the current item with the new format policy.
    /// Swapping outputs on a prepared item snaps the playhead to a keyframe, so the exact
    /// position is restored with a precise seek (which also delivers the paused frame again).
    func setPixelFormatPolicy(_ policy: FrameProvider.PixelFormatPolicy) {
        pixelFormatPolicy = policy
        guard let item = player.currentItem, let info = videoInfo else { return }
        let position = player.currentTime()
        frameProvider.attach(to: item, info: info, policy: policy)
        if position.isNumeric {
            seek(to: position.seconds, precise: true)
        }
    }

    /// Metal could not read the decoder's native format: request bi-planar 4:2:0 for this item.
    private func fallBackToExplicitFrameFormat() {
        guard let item = player.currentItem, let info = videoInfo else { return }
        let position = player.currentTime()
        frameProvider.attach(to: item, info: info, policy: .matchSource)
        if position.isNumeric {
            seek(to: position.seconds, precise: true)
        }
    }

    // MARK: - Audio & rate

    func setVolume(_ newValue: Float) {
        volume = min(max(newValue, 0), 1)
        player.volume = volume
        if volume > 0, isMuted {
            toggleMute()
        }
        UserDefaults.standard.set(volume, forKey: Keys.volume)
    }

    func adjustVolume(by delta: Float) {
        setVolume(volume + delta)
    }

    func toggleMute() {
        isMuted.toggle()
        player.isMuted = isMuted
    }

    func setRate(_ newRate: Float) {
        rate = newRate
        player.defaultRate = newRate
        if player.timeControlStatus != .paused {
            player.rate = newRate
        }
    }

    func changeRate(by steps: Int) {
        let rates = Self.availableRates
        let index = rates.firstIndex(of: rate) ?? rates.firstIndex(of: 1.0)!
        setRate(rates[min(max(index + steps, 0), rates.count - 1)])
    }

    static func label(forRate rate: Float) -> String {
        rate == rate.rounded() ? "\(Int(rate))×" : "\(rate.formatted(.number.precision(.fractionLength(1...2))))×"
    }

    // MARK: - Observation

    private func installPlayerObservers() {
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: CMTime(value: 1, timescale: 10),
            queue: .main
        ) { [weak self] time in
            MainActor.assumeIsolated {
                guard let self, !self.isScrubbing, !self.seekInFlight, time.isNumeric else { return }
                self.currentTime = time.seconds
            }
        }

        playerObservations.append(
            player.observe(\.timeControlStatus, options: [.initial, .new]) { [weak self] player, _ in
                let playing = player.timeControlStatus != .paused
                Task { @MainActor [weak self] in
                    self?.isPlaying = playing
                }
            }
        )
    }

    private func observe(_ item: AVPlayerItem) {
        itemObservations.append(
            item.observe(\.status, options: [.new]) { [weak self] item, _ in
                guard item.status == .failed else { return }
                let message = item.error?.localizedDescription ?? "Unknown playback error"
                Task { @MainActor [weak self] in
                    self?.errorMessage = "Playback failed: \(message)"
                }
            }
        )

        endObserver = NotificationCenter.default.addObserver(
            forName: AVPlayerItem.didPlayToEndTimeNotification,
            object: item,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                if self.playlist.hasNext {
                    self.next()
                }
            }
        }
    }

    private func tearDownItemObservers() {
        itemObservations.forEach { $0.invalidate() }
        itemObservations.removeAll()
        if let endObserver {
            NotificationCenter.default.removeObserver(endObserver)
            self.endObserver = nil
        }
    }
}
