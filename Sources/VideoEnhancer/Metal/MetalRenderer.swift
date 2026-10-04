import CoreGraphics
import Foundation
import Metal
import QuartzCore
import simd

/// Must match `PresentUniforms` in Render.metal.
struct PresentUniforms {
    var contentRect: SIMD4<Float>
    var split: SIMD4<Float>
    var filters: SIMD4<Int32>
}

struct ComparisonState: Sendable, Equatable {
    var enabled = false
    /// Divider position in the video, 0…1 from the left. Left = Original, right = Enhanced.
    var position: Double = 0.5
}

/// Presents processed frames into a `CAMetalLayer` on the render thread.
///
/// Redraws only when something changed (new frame, resize, comparison divider), so a paused
/// video costs no GPU time.
final class MetalRenderer: @unchecked Sendable {
    private let context: MetalContext
    private let engine: VideoProcessingEngine
    private let pipeline: MTLRenderPipelineState
    private let pixelFormat: MTLPixelFormat = .bgra8Unorm

    private let lock = NSLock()
    private var layer: CAMetalLayer?
    private var pendingDrawableSize: CGSize?
    private var comparison = ComparisonState()
    private var needsRedraw = true
    private var clearRequested = false

    // Render-thread state.
    private var current: ProcessedFrame?
    private var drawableSize: CGSize = .zero

    private let monitor: PerformanceMonitor

    init(context: MetalContext, engine: VideoProcessingEngine) {
        monitor = engine.monitor
        self.context = context
        self.engine = engine
        pipeline = context.renderPipeline(vertex: "presentVertex", fragment: "presentFragment", pixelFormat: .bgra8Unorm)
    }

    func makeLayer() -> CAMetalLayer {
        let layer = CAMetalLayer()
        layer.device = context.device
        layer.pixelFormat = pixelFormat
        layer.framebufferOnly = true
        layer.maximumDrawableCount = 3
        layer.displaySyncEnabled = true
        // Working RGB is BT.709 gamma-encoded; let the system color-match it to the display.
        layer.colorspace = CGColorSpace(name: CGColorSpace.itur_709)
        layer.backgroundColor = CGColor(gray: 0, alpha: 1)
        lock.withLock { self.layer = layer; needsRedraw = true }
        return layer
    }

    func setDrawableSize(_ size: CGSize) {
        lock.withLock {
            pendingDrawableSize = size
            needsRedraw = true
        }
    }

    func setComparison(_ state: ComparisonState) {
        lock.withLock {
            guard comparison != state else { return }
            comparison = state
            needsRedraw = true
        }
    }

    /// Shows black (no media).
    func clear() {
        lock.withLock { clearRequested = true; needsRedraw = true }
    }

    /// Size of the video area in drawable pixels for the current frame (target for upscalers).
    var currentDrawableSize: CGSize { lock.withLock { pendingDrawableSize ?? drawableSize } }

    // MARK: - Render thread

    func vsync(targetTime: CFTimeInterval, interval: CFTimeInterval) {
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "Frame"

        let (layer, newSize, state, redraw, clear): (CAMetalLayer?, CGSize?, ComparisonState, Bool, Bool) = lock.withLock {
            defer { pendingDrawableSize = nil; needsRedraw = false; clearRequested = false }
            return (self.layer, pendingDrawableSize, comparison, needsRedraw, clearRequested)
        }
        if clear { current = nil }
        if let newSize, newSize != drawableSize, newSize.width > 0, newSize.height > 0 {
            drawableSize = newSize
            layer?.drawableSize = newSize
        }
        engine.setTargetSize(drawableSize)

        var dirty = redraw
        var newFrame = false
        if let frame = engine.frame(forVsync: targetTime, interval: interval, commandBuffer: commandBuffer) {
            current = frame
            dirty = true
            newFrame = true
        }

        let monitor = self.monitor
        guard dirty, let layer, drawableSize.width > 0, let drawable = layer.nextDrawable() else {
            commandBuffer.addCompletedHandler { buffer in
                monitor.recordGPU(start: buffer.gpuStartTime, end: buffer.gpuEndTime, isRenderPass: false)
            }
            commandBuffer.commit()
            return
        }
        // Frame time of the non-AI paths: conversion + enhancement + present of a drawn frame.
        commandBuffer.addCompletedHandler { buffer in
            monitor.recordGPU(start: buffer.gpuStartTime, end: buffer.gpuEndTime, isRenderPass: true)
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = drawable.texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store

        if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) {
            encoder.label = "Present"
            if let frame = current {
                encode(frame, comparison: state, into: encoder)
            }
            encoder.endEncoding()
        }

        if let frame = current, newFrame {
            let pulledAt = frame.pulledAt, dueAt = frame.dueAt
            drawable.addPresentedHandler { presented in
                if presented.presentedTime > 0 {
                    monitor.recordPresented(pulledAt: pulledAt, dueAt: dueAt, presentedAt: presented.presentedTime)
                }
            }
        }
        commandBuffer.present(drawable)
        let shown = current
        commandBuffer.addCompletedHandler { _ in _ = shown }
        commandBuffer.commit()
    }

    private func encode(_ frame: ProcessedFrame, comparison: ComparisonState, into encoder: MTLRenderCommandEncoder) {
        let rect = AspectFit.rect(content: frame.displaySize, in: drawableSize)
        let enhanced = frame.enhanced ?? frame.original
        // Upsampling uses bicubic for the enhanced side; the Original side uses plain bilinear,
        // i.e. what a regular player shows.
        let enhancedFilter: Int32 = CGFloat(max(enhanced.width, enhanced.height)) < max(rect.width, rect.height) ? 1 : 0
        var uniforms = PresentUniforms(
            contentRect: SIMD4(Float(rect.minX), Float(rect.minY), Float(rect.width), Float(rect.height)),
            split: SIMD4(Float(comparison.position), comparison.enabled ? 1 : 0, 2, 0),
            filters: SIMD4(enhancedFilter, 0, Int32(frame.rotation), 0)
        )
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(enhanced.texture, index: 0)
        encoder.setFragmentTexture(frame.original.texture, index: 1)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<PresentUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    }
}
