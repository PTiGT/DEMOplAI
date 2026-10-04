import Foundation
import UniformTypeIdentifiers

/// Ordered list of video files used by Previous / Next.
struct Playlist: Equatable {
    private(set) var items: [URL]
    private(set) var index: Int

    init(items: [URL] = [], index: Int = 0) {
        self.items = items
        self.index = items.indices.contains(index) ? index : 0
    }

    var current: URL? { items.indices.contains(index) ? items[index] : nil }
    var hasPrevious: Bool { index > 0 }
    var hasNext: Bool { index + 1 < items.count }

    @discardableResult
    mutating func moveToNext() -> URL? {
        guard hasNext else { return nil }
        index += 1
        return current
    }

    @discardableResult
    mutating func moveToPrevious() -> URL? {
        guard hasPrevious else { return nil }
        index -= 1
        return current
    }

    /// All videos in the file's folder, in Finder order, positioned at `file`.
    static func folder(containing file: URL) -> Playlist {
        let directory = file.deletingLastPathComponent()
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.contentTypeKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        let videos = contents
            .filter(isVideoFile)
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }

        let target = file.standardizedFileURL.path
        guard let position = videos.firstIndex(where: { $0.standardizedFileURL.path == target }) else {
            return Playlist(items: [file])
        }
        return Playlist(items: videos, index: position)
    }

    static func isVideoFile(_ url: URL) -> Bool {
        let type = (try? url.resourceValues(forKeys: [.contentTypeKey]))?.contentType
            ?? UTType(filenameExtension: url.pathExtension)
        return type?.conforms(to: .movie) ?? false
    }
}
