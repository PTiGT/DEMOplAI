import Darwin
import Foundation

/// Process-level resource usage from public Darwin APIs.
enum SystemMetrics {
    /// Physical memory footprint of this process — the number Xcode's memory gauge and Activity
    /// Monitor's "Memory" column show (`task_vm_info.phys_footprint`).
    static func memoryFootprintBytes() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : nil
    }

    /// Total user + system CPU time consumed by this process so far, in seconds.
    static func cpuTimeSeconds() -> Double {
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        func seconds(_ t: timeval) -> Double { Double(t.tv_sec) + Double(t.tv_usec) / 1_000_000 }
        return seconds(usage.ru_utime) + seconds(usage.ru_stime)
    }

    static var cpuCoreCount: Int { ProcessInfo.processInfo.activeProcessorCount }

    static var thermalState: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "Nominal"
        case .fair: "Fair"
        case .serious: "Serious"
        case .critical: "Critical"
        @unknown default: "Unknown"
        }
    }
}

/// CPU usage between successive calls (100% = one core fully busy).
struct CPUUsageSampler {
    private var lastCPU = SystemMetrics.cpuTimeSeconds()
    private var lastWall = CFAbsoluteTimeGetCurrent()

    mutating func sample() -> Double? {
        let cpu = SystemMetrics.cpuTimeSeconds()
        let wall = CFAbsoluteTimeGetCurrent()
        defer { lastCPU = cpu; lastWall = wall }
        let elapsed = wall - lastWall
        return elapsed > 0.05 ? (cpu - lastCPU) / elapsed * 100 : nil
    }
}
