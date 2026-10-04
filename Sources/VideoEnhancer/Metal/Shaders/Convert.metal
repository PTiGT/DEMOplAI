// Decoded frame → working RGB (rgba16Float, BT.709 gamma-encoded, source resolution).

kernel void convertYCbCr420(texture2d<float, access::sample> lumaTex [[texture(0)]],
                            texture2d<float, access::sample> chromaTex [[texture(1)]],
                            texture2d<float, access::write> output [[texture(2)]],
                            constant YCbCrParams &params [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
    constexpr sampler linearSampler(coord::normalized, filter::linear, address::clamp_to_edge);

    float2 size = float2(output.get_width(), output.get_height());
    float2 uv = (float2(gid) + 0.5) / size;
    float y = lumaTex.read(gid).r;
    // 4:2:0 chroma is sited between luma samples horizontally (left-sited in MPEG-2/H.264),
    // bilinear sampling at the luma center is a good approximation for both sitings.
    float2 cbcr = chromaTex.sample(linearSampler, uv).rg;
    output.write(float4(ycbcrToRGB(y, cbcr, params), 1.0), gid);
}

kernel void convertBGRA(texture2d<float, access::read> input [[texture(0)]],
                        texture2d<float, access::write> output [[texture(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
    output.write(float4(input.read(gid).rgb, 1.0), gid);
}
