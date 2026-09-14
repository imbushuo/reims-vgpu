#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong src_offset;
    ulong src_pitch;
    ulong dst_offset;
    ulong dst_pitch;
    uint width;
    uint height;
    uint input;
    uint reserved;
};

kernel void seed_rgba8(
    device const uchar *source [[buffer(0)]],
    device uchar *destination [[buffer(1)]],
    constant Params &p [[buffer(2)]],
    device const uchar *half_lut [[buffer(3)]],
    uint2 position [[thread_position_in_grid]])
{
    if (position.x >= p.width || position.y >= p.height) return;
    ulong src = p.src_offset + ulong(position.y) * p.src_pitch
        + ulong(position.x) * (p.input == 2 ? 8 : 4);
    ulong dst = p.dst_offset + ulong(position.y) * p.dst_pitch + ulong(position.x) * 4;
    uchar4 color;
    if (p.input == 2) {
        for (uint channel = 0; channel < 4; ++channel) {
            uint bits = uint(source[src + channel * 2])
                | (uint(source[src + channel * 2 + 1]) << 8);
            color[channel] = half_lut[bits];
        }
    } else if (p.input == 1) {
        color = uchar4(source[src + 2], source[src + 1], source[src], source[src + 3]);
    } else {
        color = uchar4(source[src], source[src + 1], source[src + 2], source[src + 3]);
    }
    *reinterpret_cast<device uchar4 *>(destination + dst) = color;
}
