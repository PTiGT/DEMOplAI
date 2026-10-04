import CoreML
import CoreVideo
import Foundation

/// Core ML super resolution with zero-copy I/O.
///
/// Input and output are IOSurface-backed Float16 pixel buffers wrapped as `MLMultiArray`
/// (`init(pixelBuffer:shape:)`), and the output is written straight into our buffer through
/// `MLPredictionOptions.outputBackings`. Metal fills the input and reads the output in place,
/// so no pixel data crosses between CPU and GPU/ANE memory.
final class CoreMLSuperResolution: SuperResolutionModel, @unchecked Sendable {
    let info: SuperResolutionModelInfo
    let computeUnits: ComputeUnitsOption
    let inputLayout: SRBufferLayout = .planarRGBFloat16
    let outputLayout: SRBufferLayout = .planarRGBFloat16

    private let model: MLModel
    private let inputPools = PixelBufferPools(pixelFormat: kCVPixelFormatType_OneComponent16Half)
    private let outputPools = PixelBufferPools(pixelFormat: kCVPixelFormatType_OneComponent16Half)

    init(model: MLModel, info: SuperResolutionModelInfo, computeUnits: ComputeUnitsOption) {
        self.model = model
        self.info = info
        self.computeUnits = computeUnits
    }

    func makeInputBuffer(size: PixelSize) throws -> CVPixelBuffer {
        guard info.inputSizes.contains(size) else { throw SRError.unsupportedInputSize(size) }
        return try inputPools.make(width: size.width, height: size.height * 3)
    }

    func process(frame: CVPixelBuffer, context: SRFrameContext) async throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(frame)
        let height = CVPixelBufferGetHeight(frame) / 3
        let scale = info.scale
        let output = try outputPools.make(width: width * scale, height: height * scale * 3)
        let input = MLMultiArray(pixelBuffer: frame, shape: [1, 3, height as NSNumber, width as NSNumber])
        let outputArray = MLMultiArray(
            pixelBuffer: output, shape: [1, 3, (height * scale) as NSNumber, (width * scale) as NSNumber]
        )
        let options = MLPredictionOptions()
        options.outputBackings = ["output": outputArray]
        let features = try MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: input)])
        _ = try await model.prediction(from: features, options: options)
        return output
    }
}
