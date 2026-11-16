#include <metal_stdlib>
using namespace metal;

// y[n] = sum_k W[n,k] * x[k] for a 4-bit, group-quantized weight matrix (decode-time GEMV).
// Packing: two weights per byte (even k in the low nibble), stored as q + 8 so 0..15 maps to -8..7.
// Scales: one fp16 scale per 32 consecutive inputs of a row.
// Dispatch: one threadgroup of 32 threads (one SIMD-group) per output row.
kernel void gemv_q4(device const uchar *Wq     [[buffer(0)]],   // [N, K/2]
                    device const half  *scales [[buffer(1)]],   // [N, K/32]
                    device const half  *x      [[buffer(2)]],   // [K]
                    device half        *y      [[buffer(3)]],   // [N]
                    constant uint      &K      [[buffer(4)]],
                    uint row  [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]])
{
    device const uchar *wrow = Wq + row * (K / 2);
    device const half  *srow = scales + row * (K / 32);
    float acc = 0.0f;
    // Each lane handles 8 consecutive inputs (4 packed bytes) per step; 32 lanes cover 256 inputs.
    // 8 consecutive inputs never straddle a 32-input group, so one scale covers them.
    for (uint k0 = lane * 8; k0 < K; k0 += 32 * 8) {
        float s = float(srow[k0 / 32]);
        uchar4 b = *((device const uchar4 *)(wrow + k0 / 2));
        float part = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            int lo = int(b[i] & 0xF) - 8;
            int hi = int(b[i] >> 4) - 8;
            part += float(lo) * float(x[k0 + 2 * i]) + float(hi) * float(x[k0 + 2 * i + 1]);
        }
        acc += s * part;                     // dequantize once per 8 weights, not once per weight
    }
    acc = simd_sum(acc);                     // reduce the 32 lanes' partial sums in registers
    if (lane == 0) y[row] = half(acc);
}
