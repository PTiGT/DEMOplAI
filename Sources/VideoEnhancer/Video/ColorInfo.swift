import CoreMedia
import CoreVideo

/// Color signalling of a video stream or a single pixel buffer (raw Core Video / Core Media strings).
/// Stage 3 uses `matrix` and `isFullRange` to pick the YCbCr → RGB conversion.
struct ColorInfo: Sendable, Equatable {
    var matrix: String?
    var primaries: String?
    var transferFunction: String?

    init(matrix: String?, primaries: String?, transferFunction: String?) {
        self.matrix = matrix
        self.primaries = primaries
        self.transferFunction = transferFunction
    }

    init(formatDescription desc: CMFormatDescription) {
        func string(_ key: CFString) -> String? {
            CMFormatDescriptionGetExtension(desc, extensionKey: key) as? String
        }
        self.init(
            matrix: string(kCMFormatDescriptionExtension_YCbCrMatrix),
            primaries: string(kCMFormatDescriptionExtension_ColorPrimaries),
            transferFunction: string(kCMFormatDescriptionExtension_TransferFunction)
        )
    }

    init(pixelBuffer: CVPixelBuffer) {
        func string(_ key: CFString) -> String? {
            CVBufferCopyAttachment(pixelBuffer, key, nil) as? String
        }
        self.init(
            matrix: string(kCVImageBufferYCbCrMatrixKey),
            primaries: string(kCVImageBufferColorPrimariesKey),
            transferFunction: string(kCVImageBufferTransferFunctionKey)
        )
    }

    var isHDR: Bool {
        transferFunction == Self.pq || transferFunction == Self.hlg
    }

    var dynamicRangeName: String {
        switch transferFunction {
        case Self.pq?: "HDR PQ"
        case Self.hlg?: "HDR HLG"
        case nil: "SDR (untagged)"
        default: "SDR"
        }
    }

    var matrixName: String { matrix.map { Self.matrixNames[$0] ?? $0 } ?? "untagged" }
    var primariesName: String { primaries.map { Self.primariesNames[$0] ?? $0 } ?? "untagged" }

    private static let pq = kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String
    private static let hlg = kCVImageBufferTransferFunction_ITU_R_2100_HLG as String

    private static let matrixNames: [String: String] = [
        kCVImageBufferYCbCrMatrix_ITU_R_709_2 as String: "BT.709",
        kCVImageBufferYCbCrMatrix_ITU_R_601_4 as String: "BT.601",
        kCVImageBufferYCbCrMatrix_ITU_R_2020 as String: "BT.2020",
        kCVImageBufferYCbCrMatrix_SMPTE_240M_1995 as String: "SMPTE 240M",
    ]

    private static let primariesNames: [String: String] = [
        kCVImageBufferColorPrimaries_ITU_R_709_2 as String: "BT.709",
        kCVImageBufferColorPrimaries_ITU_R_2020 as String: "BT.2020",
        kCVImageBufferColorPrimaries_P3_D65 as String: "Display P3",
        kCVImageBufferColorPrimaries_EBU_3213 as String: "EBU 3213 (PAL)",
        kCVImageBufferColorPrimaries_SMPTE_C as String: "SMPTE C (NTSC)",
    ]

    /// e.g. "SDR · BT.709", "HDR PQ · BT.2020".
    var summary: String {
        "\(dynamicRangeName) · \(primaries == nil ? matrixName : primariesName)"
    }
}
