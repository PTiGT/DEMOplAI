import Foundation

enum AppResources {
    /// The SwiftPM resource bundle. In the packaged .app it lives in Contents/Resources (the generated
    /// `Bundle.module` only looks next to the executable's bundle root, which code signing forbids);
    /// for `swift run` / Xcode runs `Bundle.module` finds it in the build folder.
    static let bundle: Bundle = {
        if let url = Bundle.main.resourceURL?.appendingPathComponent("VideoEnhancer_VideoEnhancer.bundle"),
           let bundle = Bundle(url: url) {
            return bundle
        }
        return Bundle.module
    }()
}
