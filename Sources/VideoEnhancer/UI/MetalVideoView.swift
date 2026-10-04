import AppKit
import QuartzCore
import SwiftUI

/// The video surface: a CAMetalLayer-backed view driven by the screen's vsync on the render thread.
struct MetalVideoView: NSViewRepresentable {
    let renderer: MetalRenderer

    func makeNSView(context: Context) -> MetalVideoHostView {
        MetalVideoHostView(renderer: renderer)
    }

    func updateNSView(_ view: MetalVideoHostView, context: Context) {}
}

final class MetalVideoHostView: NSView {
    private let renderer: MetalRenderer
    private var vsyncLink: CADisplayLink?

    init(renderer: MetalRenderer) {
        self.renderer = renderer
        super.init(frame: .zero)
        wantsLayer = true
        layerContentsRedrawPolicy = .never
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) is not used")
    }

    override func makeBackingLayer() -> CALayer {
        renderer.makeLayer()
    }

    override func layout() {
        super.layout()
        updateDrawableSize()
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        updateDrawableSize()
    }

    private func updateDrawableSize() {
        let scale = window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2
        layer?.contentsScale = scale
        renderer.setDrawableSize(CGSize(width: (bounds.width * scale).rounded(), height: (bounds.height * scale).rounded()))
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        vsyncLink?.invalidate()
        vsyncLink = nil
        guard window != nil else { return }
        updateDrawableSize()

        // The view's display link follows the window to whichever screen it is on.
        let renderer = self.renderer
        let target = VsyncTarget { link in
            renderer.vsync(targetTime: link.targetTimestamp, interval: link.targetTimestamp - link.timestamp)
        }
        let link = displayLink(target: target, selector: #selector(VsyncTarget.tick(_:)))
        link.add(to: RenderThread.shared.runLoop, forMode: .common)
        vsyncLink = link
    }
}
