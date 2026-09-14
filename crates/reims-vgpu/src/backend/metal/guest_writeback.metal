#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong src_offset;
    ulong src_pitch;
    ulong dst_offset;
    ulong dst_pitch;
    uint width;
    uint height;
    uint half_output;
    uint reserved;
};

kernel void store_rgba8(
    device const uchar *source [[buffer(0)]],
    device uchar *destination [[buffer(1)]],
    constant Params &p [[buffer(2)]],
    constant ushort *half_lut [[buffer(3)]],
    uint2 position [[thread_position_in_grid]])
{
    if (position.x >= p.width || position.y >= p.height) return;
    ulong src = p.src_offset + ulong(position.y) * p.src_pitch + ulong(position.x) * 4;
    ulong dst = p.dst_offset + ulong(position.y) * p.dst_pitch
        + ulong(position.x) * (p.half_output ? 8 : 4);
    if (p.half_output) {
        for (uint channel = 0; channel < 4; ++channel) {
            ushort bits = half_lut[source[src + channel]];
            destination[dst + channel * 2] = uchar(bits & 255);
            destination[dst + channel * 2 + 1] = uchar(bits >> 8);
        }
    } else {
        destination[dst] = source[src + 2];
        destination[dst + 1] = source[src + 1];
        destination[dst + 2] = source[src];
        destination[dst + 3] = source[src + 3];
    }
}
