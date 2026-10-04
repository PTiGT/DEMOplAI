// Pre- and post-processing around Core ML inference, all on the GPU.

/// Must match `TensorParameters` in SuperResolutionProcessor.swift.
struct TensorParams {
    int4 size;       // x, y: tensor (model input) width/height
    float4 values;   // x: strength (consistency / stabilization)
};

// MARK: - Source RGB → model input tensor

/// Resamples the working RGB frame to the model input size and writes planar RGB (NCHW) into a
/// Float16 texture of size W × 3H that aliases the Core ML input buffer (IOSurface).
/// Downscaling averages a footprint of bilinear taps (area filter) to avoid aliasing.
kernel void rgbToPlanarTensor(texture2d<float, access::sample> source [[texture(0)]],
                              texture2d<float, access::write> tensor [[texture(1)]],
                              constant TensorParams &p [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]]) {
    int w = p.size.x, h = p.size.y;
    if (int(gid.x) >= w || int(gid.y) >= h) return;
    constexpr sampler linearSampler(coord::normalized, filter::linear, address::clamp_to_edge);

    float2 srcSize = float2(source.get_width(), source.get_height());
    float2 footprint = srcSize / float2(w, h);          // source pixels per tensor pixel
    int2 taps = clamp(int2(ceil(footprint - 0.01)), 1, 4);
    float3 sum = 0.0;
    for (int j = 0; j < taps.y; j++) {
        for (int i = 0; i < taps.x; i++) {
            float2 offset = (float2(i, j) + 0.5) / float2(taps) - 0.5;   // within the output pixel
            float2 uv = (float2(gid) + 0.5 + offset) / float2(w, h);
            sum += source.sample(linearSampler, uv).rgb;
        }
    }
    float3 rgb = saturate(sum / float(taps.x * taps.y));
    tensor.write(float4(rgb.r), uint2(gid.x, gid.y));
    tensor.write(float4(rgb.g), uint2(gid.x, gid.y + h));
    tensor.write(float4(rgb.b), uint2(gid.x, gid.y + 2 * h));
}

// MARK: - Model output tensor → RGB

kernel void planarTensorToRGB(texture2d<float, access::read> tensor [[texture(0)]],
                              texture2d<float, access::write> output [[texture(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint w = output.get_width(), h = output.get_height();
    if (gid.x >= w || gid.y >= h) return;
    float r = tensor.read(uint2(gid.x, gid.y)).r;
    float g = tensor.read(uint2(gid.x, gid.y + h)).r;
    float b = tensor.read(uint2(gid.x, gid.y + 2 * h)).r;
    output.write(float4(saturate(float3(r, g, b)), 1.0), gid);
}

// MARK: - Low-frequency consistency

/// Residual at model-input resolution: source (area-sampled) minus the AI output box-downsampled
/// to the same grid. Anything the model changed at low frequency (brightness, color, large shapes)
/// shows up here.
kernel void consistencyResidual(texture2d<float, access::sample> source [[texture(0)]],
                                texture2d<float, access::read> enhanced [[texture(1)]],
                                texture2d<float, access::write> residual [[texture(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
    uint w = residual.get_width(), h = residual.get_height();
    if (gid.x >= w || gid.y >= h) return;
    constexpr sampler linearSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    float3 lr = source.sample(linearSampler, uv).rgb;

    uint s = enhanced.get_width() / w;   // integer model scale
    float3 sum = 0.0;
    for (uint j = 0; j < s; j++) {
        for (uint i = 0; i < s; i++) {
            sum += enhanced.read(uint2(gid.x * s + i, gid.y * s + j)).rgb;
        }
    }
    residual.write(float4(lr - sum / float(s * s), 1.0), gid);
}

/// 3×3 tent blur of the residual at model-input resolution. The blur keeps 8×8 codec block edges
/// out of the correction; doing it here (low resolution) is ~16× cheaper than at output resolution.
kernel void blurResidual(texture2d<float, access::read> input [[texture(0)]],
                         texture2d<float, access::write> output [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    int w = output.get_width(), h = output.get_height();
    if (int(gid.x) >= w || int(gid.y) >= h) return;
    const float k[3] = { 0.25, 0.5, 0.25 };
    float3 sum = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            uint2 q = uint2(clamp(int(gid.x) + i, 0, w - 1), clamp(int(gid.y) + j, 0, h - 1));
            sum += input.read(q).rgb * k[i + 1] * k[j + 1];
        }
    }
    output.write(float4(sum, 1.0), gid);
}

/// Adds the blurred residual back (bicubic-upsampled): the AI keeps its high-frequency detail while
/// brightness, color and coarse structure follow the source.
kernel void consistencyApply(texture2d<float, access::read> enhanced [[texture(0)]],
                             texture2d<float, access::sample> residual [[texture(1)]],
                             texture2d<float, access::write> output [[texture(2)]],
                             constant TensorParams &p [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint w = output.get_width(), h = output.get_height();
    if (gid.x >= w || gid.y >= h) return;
    constexpr sampler linearSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    float3 r = sampleBicubic(residual, linearSampler, uv).rgb;
    float3 c = enhanced.read(gid).rgb + r * p.values.x;
    output.write(float4(saturate(c), 1.0), gid);
}

// MARK: - Temporal stabilization

/// Per source pixel: how still it is between the two frames (1 = unchanged, 0 = moving), from the
/// largest luma change in its 3×3 neighbourhood. Computed at source resolution, sampled bilinearly later.
kernel void stillnessMask(texture2d<float, access::read> sourceNow [[texture(0)]],
                          texture2d<float, access::read> sourcePrevious [[texture(1)]],
                          texture2d<float, access::write> mask [[texture(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    int w = mask.get_width(), h = mask.get_height();
    if (int(gid.x) >= w || int(gid.y) >= h) return;
    float change = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            uint2 q = uint2(clamp(int(gid.x) + i, 0, w - 1), clamp(int(gid.y) + j, 0, h - 1));
            change = max(change, abs(luma(sourceNow.read(q).rgb) - luma(sourcePrevious.read(q).rgb)));
        }
    }
    mask.write(float4(1.0 - smoothstep(1.5 / 255.0, 6.0 / 255.0, change)), gid);
}

/// Image SR runs frame by frame, so hallucinated texture can "boil". Where the source did not change
/// between frames, blend the new AI output with the previous (already stabilized) output; where
/// the source moved, use the new output only (no ghosting).
kernel void temporalStabilize(texture2d<float, access::read> current [[texture(0)]],
                              texture2d<float, access::read> previous [[texture(1)]],
                              texture2d<float, access::sample> stillness [[texture(2)]],
                              texture2d<float, access::write> output [[texture(3)]],
                              constant TensorParams &p [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint w = output.get_width(), h = output.get_height();
    if (gid.x >= w || gid.y >= h) return;
    constexpr sampler linearSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    float still = stillness.sample(linearSampler, (float2(gid) + 0.5) / float2(w, h)).r;
    float3 now = current.read(gid).rgb;
    float3 before = previous.read(gid).rgb;
    output.write(float4(mix(now, before, p.values.x * still), 1.0), gid);
}

// MARK: - Source RGB → interleaved RGBA input (models with rgbaFloat16 layout)

kernel void rgbToRGBA(texture2d<float, access::sample> source [[texture(0)]],
                      texture2d<float, access::write> output [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint w = output.get_width(), h = output.get_height();
    if (gid.x >= w || gid.y >= h) return;
    constexpr sampler linearSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    float2 footprint = float2(source.get_width(), source.get_height()) / float2(w, h);
    int2 taps = clamp(int2(ceil(footprint - 0.01)), 1, 4);
    float3 sum = 0.0;
    for (int j = 0; j < taps.y; j++) {
        for (int i = 0; i < taps.x; i++) {
            float2 offset = (float2(i, j) + 0.5) / float2(taps) - 0.5;
            sum += source.sample(linearSampler, (float2(gid) + 0.5 + offset) / float2(w, h)).rgb;
        }
    }
    output.write(float4(saturate(sum / float(taps.x * taps.y)), 1.0), gid);
}
