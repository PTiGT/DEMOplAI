// Final pass into the CAMetalLayer drawable: aspect-fit placement, optional Original | Enhanced split.

/// Must match `PresentUniforms` in MetalRenderer.swift.
struct PresentUniforms {
    float4 contentRect;   // x, y, width, height of the video in drawable pixels
    float4 split;         // x: divider position in content (0…1), y: 1 = comparison on, z: divider width px
    int4 filters;         // x: enhanced filter (0 bilinear, 1 bicubic), y: original filter, z: rotation
};

struct PresentVertexOut {
    float4 position [[position]];
};

vertex PresentVertexOut presentVertex(uint vid [[vertex_id]]) {
    // One triangle that covers the whole viewport.
    float2 pos = float2((vid << 1) & 2, vid & 2);
    PresentVertexOut out;
    out.position = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

/// Display-space uv → texture uv for a clockwise display rotation.
inline float2 rotateUV(float2 uv, int rotation) {
    switch (rotation) {
        case 90: return float2(uv.y, 1.0 - uv.x);
        case 180: return float2(1.0 - uv.x, 1.0 - uv.y);
        case 270: return float2(1.0 - uv.y, uv.x);
        default: return uv;
    }
}

/// 4×4 ordered dither, ±0.5 LSB of 8-bit, hides banding when float content is written to 8-bit.
inline float ditherOffset(uint2 p) {
    const float bayer[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
    return (bayer[(p.y & 3) * 4 + (p.x & 3)] / 16.0 - 0.5) / 255.0;
}

fragment float4 presentFragment(PresentVertexOut in [[stage_in]],
                                texture2d<float, access::sample> enhanced [[texture(0)]],
                                texture2d<float, access::sample> original [[texture(1)]],
                                constant PresentUniforms &u [[buffer(0)]]) {
    constexpr sampler linearSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    float2 p = in.position.xy;
    float2 local = (p - u.contentRect.xy) / u.contentRect.zw;
    if (any(local < 0.0) || any(local > 1.0)) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    bool showOriginal = u.split.y > 0.5 && local.x < u.split.x;
    float2 uv = rotateUV(local, u.filters.z);
    float3 rgb;
    if (showOriginal) {
        rgb = u.filters.y == 1 ? sampleBicubic(original, linearSampler, uv).rgb
                               : original.sample(linearSampler, uv).rgb;
    } else {
        rgb = u.filters.x == 1 ? sampleBicubic(enhanced, linearSampler, uv).rgb
                               : enhanced.sample(linearSampler, uv).rgb;
    }

    if (u.split.y > 0.5) {
        float dividerX = u.contentRect.x + u.split.x * u.contentRect.z;
        if (abs(p.x - dividerX) < u.split.z * 0.5) {
            rgb = float3(1.0);
        }
    }
    rgb = saturate(rgb + ditherOffset(uint2(p)));
    return float4(rgb, 1.0);
}
