import SwiftUI

/// Menu bar commands. Menu key equivalents work regardless of which control has focus.
struct PlayerCommands: Commands {
    let player: VideoPlayer
    private var playback: PlaybackController { player.playback }
    private var enhancement: EnhancementController { player.enhancement }

    @AppStorage(UserDefaultsKeys.showFrameInfo) private var showFrameInfo = false

    var body: some Commands {
        CommandGroup(after: .toolbar) {
            Toggle("Performance Monitor", isOn: $showFrameInfo)
                .keyboardShortcut("i", modifiers: .command)
            Picker("Decoded Frame Format", selection: Binding(
                get: { playback.pixelFormatPolicy },
                set: { playback.setPixelFormatPolicy($0) }
            )) {
                Text("Automatic").tag(FrameProvider.PixelFormatPolicy.automatic)
                Text("Match Source (420v / x420)").tag(FrameProvider.PixelFormatPolicy.matchSource)
                Text("Decoder Native (lossless-compressed)").tag(FrameProvider.PixelFormatPolicy.decoderNative)
            }
            Divider()
        }

        CommandMenu("Quality") {
            ForEach(Array(QualityMode.allCases.enumerated()), id: \.element) { index, mode in
                Toggle(mode.title, isOn: Binding(
                    get: { enhancement.configuration.mode == mode },
                    set: { if $0 { enhancement.configuration.mode = mode } }
                ))
                .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: .command)
            }
            Divider()
            Toggle("Dynamic Quality", isOn: Binding(
                get: { enhancement.configuration.dynamicQuality },
                set: { enhancement.configuration.dynamicQuality = $0 }
            ))
            Divider()
            Picker("Compare Original vs Enhanced", selection: Binding(
                get: { enhancement.comparisonMode },
                set: { enhancement.comparisonMode = $0 }
            )) {
                ForEach(EnhancementController.ComparisonMode.allCases) { Text($0.rawValue).tag($0) }
            }
            Divider()
            Button("Benchmark…") { enhancement.showBenchmark = true }
                .keyboardShortcut("b", modifiers: .command)
            Button("Cycle Comparison") {
                let all = EnhancementController.ComparisonMode.allCases
                enhancement.comparisonMode = all[(all.firstIndex(of: enhancement.comparisonMode)! + 1) % all.count]
            }
            .keyboardShortcut("k", modifiers: .command)
        }

        CommandGroup(replacing: .newItem) {
            Button("Open…") { playback.presentOpenPanel() }
                .keyboardShortcut("o", modifiers: .command)
        }

        CommandMenu("Playback") {
            Button("Play / Pause") { playback.togglePlayPause() }
                .keyboardShortcut(.space, modifiers: [])
                .disabled(!playback.hasMedia)

            Divider()

            Button("Jump Back 5 s") { playback.step(by: -5) }
                .keyboardShortcut(.leftArrow, modifiers: [])
                .disabled(!playback.hasMedia)
            Button("Jump Forward 5 s") { playback.step(by: 5) }
                .keyboardShortcut(.rightArrow, modifiers: [])
                .disabled(!playback.hasMedia)
            Button("Jump Back 30 s") { playback.step(by: -30) }
                .keyboardShortcut(.leftArrow, modifiers: .shift)
                .disabled(!playback.hasMedia)
            Button("Jump Forward 30 s") { playback.step(by: 30) }
                .keyboardShortcut(.rightArrow, modifiers: .shift)
                .disabled(!playback.hasMedia)

            Divider()

            Button("Previous") { playback.previous() }
                .keyboardShortcut(.leftArrow, modifiers: .command)
                .disabled(!playback.hasMedia)
            Button("Next") { playback.next() }
                .keyboardShortcut(.rightArrow, modifiers: .command)
                .disabled(!playback.playlist.hasNext)

            Divider()

            Picker("Speed", selection: Binding(get: { playback.rate }, set: { playback.setRate($0) })) {
                ForEach(PlaybackController.availableRates, id: \.self) { rate in
                    Text(PlaybackController.label(forRate: rate)).tag(rate)
                }
            }
            Button("Faster") { playback.changeRate(by: 1) }
                .keyboardShortcut("]", modifiers: [])
            Button("Slower") { playback.changeRate(by: -1) }
                .keyboardShortcut("[", modifiers: [])

            Divider()

            Button("Volume Up") { playback.adjustVolume(by: 0.1) }
                .keyboardShortcut(.upArrow, modifiers: [])
            Button("Volume Down") { playback.adjustVolume(by: -0.1) }
                .keyboardShortcut(.downArrow, modifiers: [])
            Button(playback.isMuted ? "Unmute" : "Mute") { playback.toggleMute() }
                .keyboardShortcut("m", modifiers: [])

            Divider()

            Button("Toggle Full Screen") { WindowActions.toggleFullScreen() }
                .keyboardShortcut("f", modifiers: [])
        }
    }
}

enum UserDefaultsKeys {
    static let showFrameInfo = "ui.showFrameInfo"
}

enum WindowActions {
    @MainActor
    static func toggleFullScreen() {
        (NSApp.keyWindow ?? NSApp.mainWindow ?? NSApp.windows.first)?.toggleFullScreen(nil)
    }
}
