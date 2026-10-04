import AppKit
import Foundation
import QuartzCore

/// Dedicated thread with its own run loop for vsync callbacks, so frame pulling, GPU encoding
/// and presentation never wait behind SwiftUI layout on the main thread.
final class RenderThread: Thread, @unchecked Sendable {
    static let shared: RenderThread = {
        let thread = RenderThread()
        thread.start()
        thread.ready.wait()
        return thread
    }()

    private let ready = DispatchSemaphore(value: 0)
    private(set) var runLoop: RunLoop!

    private override init() {
        super.init()
        name = "VideoEnhancer.Render"
        qualityOfService = .userInteractive
    }

    /// Runs `block` on the render thread (thread-safe; wakes the run loop).
    func perform(_ block: @escaping @Sendable () -> Void) {
        let loop = runLoop.getCFRunLoop()
        CFRunLoopPerformBlock(loop, CFRunLoopMode.commonModes.rawValue, block)
        CFRunLoopWakeUp(loop)
    }

    override func main() {
        runLoop = RunLoop.current
        // A port keeps the run loop alive before the first display link is added.
        runLoop.add(Port(), forMode: .default)
        ready.signal()
        while !isCancelled {
            _ = runLoop.run(mode: .default, before: .distantFuture)
        }
    }
}

/// Target object for `CADisplayLink`. Kept separate from the (main-actor) view because the
/// callback runs on the render thread.
final class VsyncTarget: NSObject, @unchecked Sendable {
    private let handler: @Sendable (CADisplayLink) -> Void

    init(handler: @escaping @Sendable (CADisplayLink) -> Void) {
        self.handler = handler
    }

    @objc nonisolated func tick(_ link: CADisplayLink) {
        handler(link)
    }
}
