// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VideoEnhancer",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(
            name: "VideoEnhancer",
            path: "Sources/VideoEnhancer",
            // Shader sources are compiled at runtime (MTLDevice.makeLibrary(source:)), so they ship as-is.
            resources: [.copy("Metal/Shaders")]
        )
    ]
)
