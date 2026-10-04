// Compiles Sources/VideoEnhancer/Metal/Shaders/*.metal the same way the app does (Common first)
// and prints compiler diagnostics. Usage: swift Tools/check_shaders.swift
import Foundation
import Metal

let folder = URL(fileURLWithPath: "Sources/VideoEnhancer/Metal/Shaders")
let files = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
    .filter { $0.pathExtension == "metal" }
    .sorted { ($0.lastPathComponent == "Common.metal" ? "" : $0.lastPathComponent) < ($1.lastPathComponent == "Common.metal" ? "" : $1.lastPathComponent) }
let source = try files.map { "// ---- \($0.lastPathComponent)\n" + (try String(contentsOf: $0, encoding: .utf8)) }.joined(separator: "\n")
let options = MTLCompileOptions()
options.mathMode = .fast
options.languageVersion = .version3_1
do {
    let library = try MTLCreateSystemDefaultDevice()!.makeLibrary(source: source, options: options)
    print("OK: \(files.map(\.lastPathComponent).joined(separator: ", "))")
    print("Functions: \(library.functionNames.sorted().joined(separator: ", "))")
} catch {
    print(error.localizedDescription)
    exit(1)
}
