// Classical (non-AI) GPU enhancement: artifact reduction, Lanczos upscale, adaptive sharpening.
// Used by the "Fast" quality mode and as the fallback when AI cannot keep up.

/// Must match `EnhanceParameters` in EnhancementPipeline.swift.
struct EnhanceParams {
    float4 values;   // x: denoise strength 0…1, y: sharpness 0…1, z: deband strength 0…1
};

// MARK: - Artifact reduction (source resolution)

/// Edge-preserving 5×5 bilateral filter. Smooths mosquito noise and blocking in flat areas while
/// keeping edges; the range weight uses luma difference so color edges survive too.
kernel void reduceArtifacts(texture2d<float, access::read> input [[texture(0)]],
                            texture2d<float, access::write> output [[texture(1)]],
                            constant EnhanceParams &p [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint w = output.get_width(), h = output.get_height();
    if (gid.x >= w || gid.y >= h) return;
    float4 center = input.read(gid);
    float strength = p.values.x;
    if (strength <= 0.001) { output.write(center, gid); return; }

    float centerLuma = luma(center.rgb);
    float rangeSigma = mix(0.015, 0.07, strength);
    float invRange = -0.5 / (rangeSigma * rangeSigma);
    const float invSpatial = -0.5 / (1.6 * 1.6);

    float3 sum = 0.0;
    float weightSum = 0.0;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            uint2 q = uint2(clamp(int(gid.x) + dx, 0, int(w) - 1), clamp(int(gid.y) + dy, 0, int(h) - 1));
            float3 c = input.read(q).rgb;
            float d = luma(c) - centerLuma;
            float wgt = exp(float(dx * dx + dy * dy) * invSpatial + d * d * invRange);
            sum += c * wgt;
            weightSum += wgt;
        }
    }
    float3 filtered = sum / weightSum;
    output.write(float4(mix(center.rgb, filtered, strength), 1.0), gid);
}

// MARK: - Lanczos upscale

inline float lanczos2(float x) {
    x = abs(x);
    if (x < 1e-5) return 1.0;
    if (x >= 2.0) return 0.0;
    float px = M_PI_F * x;
    return 2.0 * sin(px) * sin(px * 0.5) / (px * px);
}

/// Lanczos-2 (4×4 taps) resampling to the output size, with anti-ringing: the result is clamped
/// to the range of the 2×2 nearest source pixels, which removes halos around hard edges.
kernel void upscaleLanczos(texture2d<float, access::read> input [[texture(0)]],
                           texture2d<float, access::write> output [[texture(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint ow = output.get_width(), oh = output.get_height();
    if (gid.x >= ow || gid.y >= oh) return;
    int iw = input.get_width(), ih = input.get_height();
    float2 scale = float2(iw, ih) / float2(ow, oh);
    float2 src = (float2(gid) + 0.5) * scale - 0.5;
    float2 base = floor(src);
    float2 f = src - base;

    float4 wx, wy;
    for (int i = 0; i < 4; i++) {
        wx[i] = lanczos2(f.x - float(i - 1));
        wy[i] = lanczos2(f.y - float(i - 1));
    }
    wx /= (wx.x + wx.y + wx.z + wx.w);
    wy /= (wy.x + wy.y + wy.z + wy.w);

    float3 sum = 0.0;
    float3 lo = 1e9, hi = -1e9;
    for (int j = 0; j < 4; j++) {
        int y = clamp(int(base.y) + j - 1, 0, ih - 1);
        float3 row = 0.0;
        for (int i = 0; i < 4; i++) {
            int x = clamp(int(base.x) + i - 1, 0, iw - 1);
            float3 c = input.read(uint2(x, y)).rgb;
            row += c * wx[i];
            if ((i == 1 || i == 2) && (j == 1 || j == 2)) { lo = min(lo, c); hi = max(hi, c); }
        }
        sum += row * wy[j];
    }
    output.write(float4(clamp(sum, lo, hi), 1.0), gid);
}

// MARK: - Contrast-adaptive sharpening (output resolution)

/// Sharpens less where local contrast is already high (no oversharpened halos) and more in
/// low-contrast detail. Same idea as AMD FidelityFX CAS, written from the published description.
kernel void sharpenAdaptive(texture2d<float, access::read> input [[texture(0)]],
                            texture2d<float, access::write> output [[texture(1)]],
                            constant EnhanceParams &p [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint w = output.get_width(), h = output.get_height();
    if (gid.x >= w || gid.y >= h) return;
    float3 e = input.read(gid).rgb;
    if (p.values.y <= 0.001) { output.write(float4(e, 1.0), gid); return; }

    uint2 maxP = uint2(w - 1, h - 1);
    float3 b = input.read(uint2(gid.x, gid.y > 0 ? gid.y - 1 : 0)).rgb;
    float3 d = input.read(uint2(gid.x > 0 ? gid.x - 1 : 0, gid.y)).rgb;
    float3 f = input.read(uint2(min(gid.x + 1, maxP.x), gid.y)).rgb;
    float3 hh = input.read(uint2(gid.x, min(gid.y + 1, maxP.y))).rgb;

    float3 mn = min(min(min(b, d), min(f, hh)), e);
    float3 mx = max(max(max(b, d), max(f, hh)), e);
    float3 amp = sqrt(saturate(min(mn, 1.0 - mx) / max(mx, 1e-4)));
    float peak = -1.0 / mix(8.0, 4.5, p.values.y);
    float3 wgt = amp * peak;
    float3 result = (b * wgt + d * wgt + f * wgt + hh * wgt + e) / (1.0 + 4.0 * wgt);
    output.write(float4(saturate(result), 1.0), gid);
}
