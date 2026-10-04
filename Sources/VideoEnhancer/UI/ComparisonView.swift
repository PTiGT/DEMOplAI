import SwiftUI

/// Original | Enhanced comparison overlay. The split itself is drawn by the Metal renderer
/// (same frame, same timestamp on both sides); this view adds the labels and the drag handle.
struct ComparisonView: View {
    @Bindable var enhancement: EnhancementController
    /// Aspect ratio of the displayed video, to place the divider inside the picture.
    let videoAspect: CGFloat

    var body: some View {
        GeometryReader { proxy in
            let rect = videoRect(in: proxy.size)
            let x = rect.minX + rect.width * enhancement.comparison.position
            ZStack(alignment: .topLeading) {
                label("Original").position(x: max(rect.minX + 50, x - 60), y: rect.minY + 24)
                label("Enhanced · \(enhancement.snapshot.pathLabel)")
                    .position(x: min(rect.maxX - 110, x + 110), y: rect.minY + 24)

                if enhancement.comparisonMode == .slider {
                    Circle()
                        .fill(.white)
                        .frame(width: 28, height: 28)
                        .overlay(Image(systemName: "arrow.left.and.right").font(.caption.bold()).foregroundStyle(.black))
                        .shadow(radius: 3)
                        .position(x: x, y: rect.midY)
                    Rectangle()
                        .fill(Color.clear)
                        .contentShape(Rectangle())
                        .frame(width: 44, height: rect.height)
                        .position(x: x, y: rect.midY)
                        .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                            let position = (value.location.x - rect.minX) / max(rect.width, 1)
                            enhancement.comparison.position = min(max(position, 0.02), 0.98)
                        })
                        .onHover { inside in inside ? NSCursor.resizeLeftRight.push() : NSCursor.pop() }
                }
            }
        }
    }

    private func videoRect(in size: CGSize) -> CGRect {
        AspectFit.rect(content: CGSize(width: videoAspect, height: 1), in: size)
    }

    private func label(_ text: String) -> some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .lineLimit(1)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.black.opacity(0.55), in: Capsule())
            .foregroundStyle(.white)
    }
}
