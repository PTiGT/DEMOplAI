import Foundation

/// Headless benchmark: `VideoEnhancer --benchmark [options]`, runs without a window and exits.
///
///   --model    model id(s), comma separated, "metal" for the Metal Fast path, or "all"   (default: all)
///   --units    cpuAndNeuralEngine | all | cpuAndGPU | cpuOnly, comma separated          (default: cpuAndNeuralEngine)
///   --input    WxH list, e.g. 640x360,1280x720                                          (default: 640x360,960x540,1280x720)
///   --duration seconds per run (default 5) · --fps arrival rate (default 60) · --inflight 1|2 (default 2, as playback)
///   --file     decode frames from this video · --no-post disable consistency/stabilization
///   --json     write all results to this file
enum CommandLineBenchmark {
    static func start() {
        Task { @MainActor in
            let code = await run(arguments: CommandLine.arguments)
            exit(code)
        }
    }

    @MainActor
    private static func run(arguments: [String]) async -> Int32 {
        func value(_ name: String) -> String? {
            guard let index = arguments.firstIndex(of: name), index + 1 < arguments.count else { return nil }
            return arguments[index + 1]
        }
        guard case .success(let context) = MetalContext.shared else {
            print("Metal unavailable"); return 1
        }
        let manager = ModelManager()
        await manager.discover()
        for problem in manager.discoveryErrors { print("model error: \(problem)") }

        var modelIDs = (value("--model") ?? "all").split(separator: ",").map(String.init)
        if modelIDs == ["all"] { modelIDs = manager.entries.map(\.id) + ["metal"] }
        let units = (value("--units") ?? "cpuAndNeuralEngine").split(separator: ",").compactMap { ComputeUnitsOption(rawValue: String($0)) }
        let inputs = (value("--input") ?? "640x360,960x540,1280x720").split(separator: ",").compactMap { token -> PixelSize? in
            let p = token.split(separator: "x").compactMap { Int($0) }
            return p.count == 2 ? PixelSize(width: p[0], height: p[1]) : nil
        }
        var base = BenchmarkConfiguration()
        base.duration = value("--duration").flatMap(Double.init) ?? 5
        base.arrivalFPS = value("--fps").flatMap(Double.init) ?? 60
        base.framesInFlight = value("--inflight").flatMap(Int.init) ?? 2
        base.sourceFile = value("--file").map { URL(fileURLWithPath: $0) }
        if arguments.contains("--no-post") { base.postprocessing = AIPostSettings(consistency: false, stabilization: false, sharpen: 0) }
        if arguments.contains("--no-stabilization") { base.postprocessing.stabilization = false }
        if arguments.contains("--no-consistency") { base.postprocessing.consistency = false }
        base.dumpDirectory = value("--dump").map { URL(fileURLWithPath: $0) }

        let runner = BenchmarkRunner(context: context)
        var results: [BenchmarkResult] = []
        print("model                          units      input      fps(avg)  processed/offered  dropped  infer med/p95 ms  total ms  pre/post GPU ms  peak MB  CPU%  GPU%")
        for id in modelIDs {
            for unit in (id == "metal" ? [ComputeUnitsOption.cpuAndNeuralEngine] : units) {
                var model: CoreMLSuperResolution?
                var loadSeconds: Double?
                if id != "metal" {
                    let start = CFAbsoluteTimeGetCurrent()
                    do {
                        model = try await manager.load(id: id, units: unit)
                        loadSeconds = CFAbsoluteTimeGetCurrent() - start
                    } catch {
                        print("\(id) \(unit.shortTitle): load failed: \(error.localizedDescription)")
                        continue
                    }
                }
                for input in inputs {
                    if let model, !model.info.inputSizes.contains(input) { continue }
                    var config = base
                    config.modelID = id == "metal" ? nil : id
                    config.computeUnits = unit
                    config.inputSize = input
                    config.metalOutputSize = input.scaled(by: 2)
                    let result = await runner.run(config, model: model, loadSeconds: loadSeconds)
                    results.append(result)
                    print(format(result))
                }
                if id != "metal" { manager.unload(id: id) }
            }
        }
        if let path = value("--json") {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            if let data = try? encoder.encode(results) {
                try? data.write(to: URL(fileURLWithPath: path))
                print("saved \(path)")
            }
        }
        return 0
    }

    private static func format(_ r: BenchmarkResult) -> String {
        func f(_ v: Double?, _ digits: Int = 1) -> String { v.map { String(format: "%.\(digits)f", $0) } ?? "—" }
        let name = (r.configuration.modelID ?? "metal").padding(toLength: 30, withPad: " ", startingAt: 0)
        let units = (r.configuration.modelID == nil ? "GPU" : r.configuration.computeUnits.shortTitle).padding(toLength: 10, withPad: " ", startingAt: 0)
        let input = r.configuration.inputSize.description.padding(toLength: 10, withPad: " ", startingAt: 0)
        var line = "\(name) \(units) \(input) \(f(r.averageFPS).padding(toLength: 9, withPad: " ", startingAt: 0))"
        line += " \("\(r.framesProcessed)/\(r.framesOffered)".padding(toLength: 18, withPad: " ", startingAt: 0))"
        line += " \(String(r.framesDropped).padding(toLength: 8, withPad: " ", startingAt: 0))"
        line += " \("\(f(r.inferenceMedianMs))/\(f(r.inferenceP95Ms))".padding(toLength: 17, withPad: " ", startingAt: 0))"
        line += " \(f(r.totalLatencyMedianMs).padding(toLength: 9, withPad: " ", startingAt: 0))"
        line += " \("\(f(r.preprocessGPUMs, 2))/\(f(r.postprocessGPUMs, 2))".padding(toLength: 16, withPad: " ", startingAt: 0))"
        line += " \(String(r.peakMemoryBytes / 1_048_576).padding(toLength: 8, withPad: " ", startingAt: 0))"
        line += " \(f(r.cpuPercent, 0).padding(toLength: 5, withPad: " ", startingAt: 0)) \(f(r.gpuBusyPercent, 0))"
        if let error = r.error { line += "  ERROR: \(error)" }
        return line
    }
}
