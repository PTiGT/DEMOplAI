import SwiftUI

struct PlayerControls: View {
    let player: VideoPlayer
    private var playback: PlaybackController { player.playback }
    private var enhancement: EnhancementController { player.enhancement }

    @State private var showQuality = false
    @State private var showSettings = false

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 14) {
                transportButtons
                timeline
                volumeControls
                speedMenu
                qualityButtons
                Button { WindowActions.toggleFullScreen() } label: {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                }
                .accessibilityLabel("Toggle Full Screen")
                .help("Full Screen (F)")
            }

            HStack(spacing: 8) {
                Text(playback.title)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let info = playback.videoInfo {
                    Text(info.summary)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer(minLength: 16)
                boostStatus
            }
            .font(.caption)
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .environment(\.colorScheme, .dark)
        .frame(maxWidth: 960)
        .padding(16)
    }

    private var transportButtons: some View {
        HStack(spacing: 12) {
            Button { playback.previous() } label: {
                Image(systemName: "backward.end.fill")
            }
            .accessibilityLabel("Previous")
            .help("Previous (⌘←)")

            Button { playback.togglePlayPause() } label: {
                Image(systemName: playback.isPlaying ? "pause.fill" : "play.fill")
                    .font(.title2)
                    .frame(width: 26)
            }
            .accessibilityLabel(playback.isPlaying ? "Pause" : "Play")
            .help("Play / Pause (Space)")

            Button { playback.next() } label: {
                Image(systemName: "forward.end.fill")
            }
            .accessibilityLabel("Next")
            .disabled(!playback.playlist.hasNext)
            .help("Next (⌘→)")
        }
    }

    private var timeline: some View {
        HStack(spacing: 8) {
            Text(TimeFormat.string(playback.currentTime, total: playback.duration))
            Slider(
                value: Binding(get: { playback.currentTime }, set: { playback.scrub(to: $0) }),
                in: 0...max(playback.duration, 0.001),
                onEditingChanged: { editing in
                    editing ? playback.beginScrubbing() : playback.endScrubbing()
                }
            )
            .controlSize(.small)
            .disabled(playback.duration <= 0)
            Text(TimeFormat.string(playback.duration, total: playback.duration))
        }
        .font(.callout.monospacedDigit())
    }

    private var volumeControls: some View {
        HStack(spacing: 6) {
            Button { playback.toggleMute() } label: {
                Image(systemName: volumeSymbol)
                    .frame(width: 20, alignment: .leading)
            }
            .accessibilityLabel(playback.isMuted ? "Unmute" : "Mute")
            .help(playback.isMuted ? "Unmute (M)" : "Mute (M)")

            Slider(
                value: Binding(get: { Double(playback.volume) }, set: { playback.setVolume(Float($0)) }),
                in: 0...1
            )
            .controlSize(.mini)
            .frame(width: 80)
        }
    }

    private var speedMenu: some View {
        Menu {
            Picker("Playback Speed", selection: Binding(get: { playback.rate }, set: { playback.setRate($0) })) {
                ForEach(PlaybackController.availableRates, id: \.self) { rate in
                    Text(PlaybackController.label(forRate: rate)).tag(rate)
                }
            }
            .pickerStyle(.inline)
        } label: {
            Text(PlaybackController.label(forRate: playback.rate))
                .font(.callout.monospacedDigit())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Playback Speed ([ / ])")
    }

    private var qualityButtons: some View {
        HStack(spacing: 12) {
            Button { showQuality.toggle() } label: {
                Image(systemName: enhancement.aiBoostOn ? "sparkles" : "sparkle")
                    .foregroundStyle(enhancement.aiBoostOn ? Color.accentColor : .primary)
            }
            .help("Quality Boost")
            .accessibilityLabel("Quality Boost")
            .popover(isPresented: $showQuality, arrowEdge: .top) { QualityMenuView(enhancement: enhancement) }

            Menu {
                Picker("Compare", selection: Binding(get: { enhancement.comparisonMode }, set: { enhancement.comparisonMode = $0 })) {
                    ForEach(EnhancementController.ComparisonMode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.inline)
            } label: {
                Label("Compare Original and Enhanced", systemImage: "rectangle.split.2x1")
                    .labelStyle(.iconOnly)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("Original vs Enhanced")
            .accessibilityLabel("Compare Original and Enhanced")

            Button { showSettings.toggle() } label: { Image(systemName: "gearshape") }
                .help("Advanced")
                .accessibilityLabel("Advanced Settings")
                .popover(isPresented: $showSettings, arrowEdge: .top) { SettingsView(player: player) }
        }
    }

    /// "AI BOOST: ON/OFF" — reflects what actually processed the frames on screen, not just the setting.
    private var boostStatus: some View {
        let s = enhancement.snapshot
        let on = enhancement.aiBoostOn
        return HStack(spacing: 6) {
            Text(on ? "AI BOOST: ON" : "AI BOOST: OFF")
                .fontWeight(.semibold)
                .foregroundStyle(on ? Color.accentColor : .secondary)
            Text("·").foregroundStyle(.secondary)
            Text(on ? (s.modelName ?? "AI") : (s.pathLabel == "—" ? enhancement.configuration.mode.title : s.pathLabel))
                .foregroundStyle(.secondary)
            if let reason = s.pathReason {
                Text("(\(reason))").foregroundStyle(.orange).lineLimit(1)
            } else if !enhancement.configuration.mode.usesAI || enhancement.modelStatus.hasPrefix("Not loaded") {
                Text("· AI Model: \(enhancement.modelStatus)").foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .help(enhancement.modelStatus)
    }

    private var volumeSymbol: String {
        if playback.isMuted || playback.volume == 0 { return "speaker.slash.fill" }
        if playback.volume < 0.34 { return "speaker.wave.1.fill" }
        if playback.volume < 0.67 { return "speaker.wave.2.fill" }
        return "speaker.wave.3.fill"
    }
}

enum TimeFormat {
    /// `m:ss`, or `h:mm:ss` when the media is an hour or longer.
    static func string(_ seconds: Double, total: Double) -> String {
        let value = seconds.isFinite ? max(Int(seconds.rounded(.down)), 0) : 0
        let (h, m, s) = (value / 3600, value / 60 % 60, value % 60)
        if total >= 3600 {
            return String(format: "%d:%02d:%02d", h, m, s)
        }
        return String(format: "%d:%02d", h * 60 + m, s)
    }
}
