import AppKit
import SwiftUI

struct PlayerView: View {
    let player: VideoPlayer
    private var playback: PlaybackController { player.playback }

    @State private var controlsVisible = true
    @State private var isHoveringControls = false
    @State private var isDropTargeted = false
    @State private var hideTask: Task<Void, Never>?
    @AppStorage(UserDefaultsKeys.showFrameInfo) private var showFrameInfo = false

    private var showsControls: Bool {
        controlsVisible || !playback.isPlaying || !playback.hasMedia
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            Color.black

            if let renderer = player.renderer {
                MetalVideoView(renderer: renderer)
            }

            if playback.hasMedia {
                Color.clear
                    .contentShape(Rectangle())
                    .onTapGesture(count: 2) { WindowActions.toggleFullScreen() }
            } else {
                EmptyPlayerView(isDropTargeted: isDropTargeted) { playback.presentOpenPanel() }
            }

            if playback.hasMedia, player.enhancement.comparisonMode != .off, let info = playback.videoInfo {
                ComparisonView(enhancement: player.enhancement,
                               videoAspect: CGFloat(info.width) / CGFloat(max(info.height, 1)))
            }

            if playback.hasMedia, showFrameInfo {
                PerformanceView(player: player)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .padding(.top, 36)
                    .padding(.leading, 16)
                    .allowsHitTesting(false)
            }

            if let message = playback.errorMessage ?? player.setupError {
                ErrorBanner(message: message) { playback.errorMessage = nil }
                    .frame(maxHeight: .infinity, alignment: .top)
                    .padding(.top, 36)
            }

            if playback.hasMedia, showsControls {
                PlayerControls(player: player)
                    .onHover { isHoveringControls = $0 }
                    .transition(.opacity)
            }
        }
        .ignoresSafeArea()
        .navigationTitle(playback.title)
        .onContinuousHover { phase in
            if case .active = phase { registerActivity() }
        }
        .onChange(of: playback.isPlaying) { registerActivity() }
        .sheet(isPresented: Binding(get: { player.enhancement.showBenchmark }, set: { player.enhancement.showBenchmark = $0 })) {
            BenchmarkView(player: player, benchmark: player.enhancement.benchmark)
        }
        .dropDestination(for: URL.self) { urls, _ in
            playback.open(urls)
            return true
        } isTargeted: { isDropTargeted = $0 }
    }

    /// Shows the controls and hides them again after a pause in mouse movement while playing.
    private func registerActivity() {
        if !controlsVisible {
            withAnimation(.easeOut(duration: 0.15)) { controlsVisible = true }
        }
        hideTask?.cancel()
        hideTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(2.5))
            guard !Task.isCancelled, playback.isPlaying, !isHoveringControls else { return }
            withAnimation(.easeInOut(duration: 0.4)) { controlsVisible = false }
            NSCursor.setHiddenUntilMouseMoves(true)
        }
    }
}

private struct EmptyPlayerView: View {
    let isDropTargeted: Bool
    let openFile: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "film.stack")
                .font(.system(size: 48, weight: .light))
                .foregroundStyle(.secondary)
            Text("Drop a video here")
                .font(.title2.weight(.medium))
            Text("MP4, MOV, M4V and other formats supported by AVFoundation")
                .font(.callout)
                .foregroundStyle(.secondary)
            Button("Open File…", action: openFile)
                .controlSize(.large)
                .keyboardShortcut(.defaultAction)
        }
        .padding(48)
        .background {
            RoundedRectangle(cornerRadius: 20)
                .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [8, 6]))
                .foregroundStyle(isDropTargeted ? Color.accentColor : Color.white.opacity(0.15))
        }
        .environment(\.colorScheme, .dark)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct ErrorBanner: View {
    let message: String
    let dismiss: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.yellow)
            Text(message)
                .lineLimit(3)
            Button(action: dismiss) {
                Image(systemName: "xmark")
            }
            .buttonStyle(.plain)
        }
        .font(.callout)
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .environment(\.colorScheme, .dark)
        .padding(.horizontal, 24)
    }
}
