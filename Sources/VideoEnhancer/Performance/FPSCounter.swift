import Foundation
import QuartzCore

/// Events per second over a sliding window of timestamps (host time, seconds). Not thread-safe;
/// owned by `PerformanceMonitor`, which serializes access.
struct FPSCounter {
    private var times: [CFTimeInterval] = []
    private let window: CFTimeInterval

    init(window: CFTimeInterval = 1.0) {
        self.window = window
    }

    mutating func record(_ time: CFTimeInterval = CACurrentMediaTime()) {
        times.append(time)
        trim(now: time)
    }

    /// Events in the window ending at `now`, per second (a stall shows up as a falling rate).
    func rate(now: CFTimeInterval = CACurrentMediaTime()) -> Double {
        Double(times.filter { now - $0 <= window && $0 <= now + 0.5 }.count) / window
    }

    mutating func reset() { times.removeAll() }

    private mutating func trim(now: CFTimeInterval) {
        if let index = times.firstIndex(where: { now - $0 <= window * 2 }), index > 0 {
            times.removeFirst(index)
        }
    }
}

/// Fixed-size window of samples with median / p95 / mean.
struct SampleWindow {
    private var values: [Double] = []
    private let capacity: Int

    init(capacity: Int = 120) {
        self.capacity = capacity
    }

    mutating func add(_ value: Double) {
        values.append(value)
        if values.count > capacity { values.removeFirst(values.count - capacity) }
    }

    mutating func reset() { values.removeAll() }

    var isEmpty: Bool { values.isEmpty }
    var count: Int { values.count }
    var last: Double? { values.last }
    var mean: Double? { values.isEmpty ? nil : values.reduce(0, +) / Double(values.count) }
    var median: Double? { percentile(0.5) }
    var p95: Double? { percentile(0.95) }

    func percentile(_ p: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        return sorted[min(sorted.count - 1, Int((Double(sorted.count - 1) * p).rounded()))]
    }
}
