// Shared types and helpers. Files in this folder are concatenated in a fixed order
// (Common first) and compiled at runtime by MetalContext.

#include <metal_stdlib>
using namespace metal;

// MARK: - YCbCr → RGB

/// Must match `YCbCrParameters` in MetalContext.swift (all members 16-byte aligned).
struct YCbCrParams {
    float4x4 toRGB;      // rows applied to (Y, Cb, Cr, 1) after range expansion
    float4 scale;        // (yScale, cScale, cScale, 0): code value → normalized
    float4 offset;       // (yOffset, cOffset, cOffset, 0)
    int4 flags;          // x: transfer (0 SDR, 1 PQ, 2 HLG), y: BT.2020 primaries → BT.709
};

// BT.2020 → BT.709 primaries, linear light.
constant float3x3 kBT2020to709 = float3x3(
    float3( 1.6605, -0.1246, -0.0182),
    float3(-0.5876,  1.1329, -0.1006),
    float3(-0.0728, -0.0083,  1.1187));

inline float3 pqToLinearNits(float3 e) {
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float3 p = pow(max(e, 0.0), 1.0 / m2);
    return 10000.0 * pow(max(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1);
}

inline float3 hlgToSceneLinear(float3 e) {
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    float3 low = e * e / 3.0;
    float3 high = (exp((e - c) / a) + b) / 12.0;
    return select(high, low, e <= 0.5);
}

/// Basic HDR → SDR: luminance-preserving extended Reinhard, then BT.1886 (gamma 2.4) encoding.
/// `linear` is relative to SDR reference white (1.0 = 100 nits-ish).
inline float3 toneMapToSDR(float3 linear, float peak) {
    float l = dot(linear, float3(0.2126, 0.7152, 0.0722));
    float mapped = l * (1.0 + l / (peak * peak)) / (1.0 + l);
    float3 rgb = linear * (l > 0.0 ? mapped / l : 0.0);
    return pow(saturate(rgb), 1.0 / 2.4);
}

inline float3 ycbcrToRGB(float y, float2 cbcr, constant YCbCrParams &p) {
    float3 ycc = float3(y, cbcr) * p.scale.xyz + p.offset.xyz;
    float3 rgb = (p.toRGB * float4(ycc, 1.0)).xyz;

    if (p.flags.x == 1) {           // PQ: absolute nits, reference white 203 nits
        float3 lin = pqToLinearNits(saturate(rgb)) / 203.0;
        if (p.flags.y == 1) lin = kBT2020to709 * lin;
        return toneMapToSDR(max(lin, 0.0), 1000.0 / 203.0);
    }
    if (p.flags.x == 2) {           // HLG: OOTF with system gamma 1.2, reference white ≈ 0.203
        float3 scene = hlgToSceneLinear(saturate(rgb));
        float ys = dot(scene, float3(0.2627, 0.6780, 0.0593));
        float3 lin = scene * pow(max(ys, 1e-6), 0.2) / 0.203;
        if (p.flags.y == 1) lin = kBT2020to709 * lin;
        return toneMapToSDR(max(lin, 0.0), 1.0 / 0.203);
    }
    if (p.flags.y == 1) {           // SDR BT.2020: convert primaries in (approximately) linear light
        float3 lin = pow(saturate(rgb), 2.4);
        return pow(saturate(kBT2020to709 * lin), 1.0 / 2.4);
    }
    return saturate(rgb);
}

// MARK: - Filtering helpers

/// Catmull-Rom weights for a fractional offset t in [0, 1).
inline float4 catmullRomWeights(float t) {
    float t2 = t * t, t3 = t2 * t;
    return float4(
        -0.5 * t3 + t2 - 0.5 * t,
         1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
         0.5 * t3 - 0.5 * t2);
}

/// Bicubic (Catmull-Rom) sample using 16 point reads. `uv` in normalized coordinates.
inline float4 sampleBicubic(texture2d<float, access::sample> tex, sampler s, float2 uv) {
    float2 size = float2(tex.get_width(), tex.get_height());
    float2 pos = uv * size - 0.5;
    float2 base = floor(pos);
    float2 f = pos - base;
    float4 wx = catmullRomWeights(f.x);
    float4 wy = catmullRomWeights(f.y);
    float4 result = 0.0;
    for (int j = 0; j < 4; j++) {
        float4 row = 0.0;
        for (int i = 0; i < 4; i++) {
            float2 p = (base + float2(i - 1, j - 1) + 0.5) / size;
            row += tex.sample(s, p) * wx[i];
        }
        result += row * wy[j];
    }
    return result;
}

inline float luma(float3 rgb) {
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}
