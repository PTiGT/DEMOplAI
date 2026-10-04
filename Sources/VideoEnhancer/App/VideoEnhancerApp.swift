import AppKit
import SwiftUI

@main
enum EntryPoint {
    static func main() {
        if CommandLine.arguments.contains("--benchmark") {
            CommandLineBenchmark.start()
            dispatchMain()
        } else {
            VideoEnhancerApp.main()
        }
    }
}

struct VideoEnhancerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Window("Video Enhancer", id: "player") {
            PlayerView(player: appDelegate.player)
                .frame(minWidth: 640, minHeight: 400)
        }
        .defaultSize(width: 1280, height: 760)
        .windowStyle(.hiddenTitleBar)
        .commands {
            PlayerCommands(player: appDelegate.player)
        }
    }
}

/// Owns app-lifetime objects so that Finder "Open With" events, which can arrive
/// before the window exists, reach the same playback controller as the UI.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let player = VideoPlayer()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Needed when the binary is started directly (`swift run`) instead of from the .app bundle.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate()
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        player.playback.open(urls)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
