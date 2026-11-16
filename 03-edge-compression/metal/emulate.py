"""Lane-by-lane NumPy emulation of gemv_q4.metal, checked against a dequantized float reference.
This validates the packing, indexing and reduction logic. It does NOT compile or time the kernel on a GPU."""
import numpy as np

def pack_q4(W, group=32):
    N, K = W.shape
    Wg = W.reshape(N, K // group, group)
    s = np.maximum(np.abs(Wg).max(-1) / 7, 1e-8)                       # symmetric, qmax = 7
    q = np.clip(np.round(Wg / s[..., None]), -8, 7).astype(np.int8).reshape(N, K)
    u = (q + 8).astype(np.uint8)
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)        # even k -> low nibble
    return packed, s.astype(np.float16), (q.reshape(N, K // group, group) * s[..., None].astype(np.float16)).reshape(N, K)

def kernel_emulation(packed, scales, x, K):
    N = packed.shape[0]; y = np.zeros(N, np.float16); x = x.astype(np.float16)
    for row in range(N):                                               # one threadgroup per row
        wrow, srow = packed[row], scales[row]; lanes = np.zeros(32, np.float32)
        for lane in range(32):
            acc = np.float32(0)
            for k0 in range(lane * 8, K, 256):
                s = np.float32(srow[k0 // 32]); b = wrow[k0 // 2: k0 // 2 + 4]; part = np.float32(0)
                for i in range(4):
                    lo = int(b[i] & 0xF) - 8; hi = int(b[i] >> 4) - 8
                    part += np.float32(lo) * np.float32(x[k0 + 2 * i]) + np.float32(hi) * np.float32(x[k0 + 2 * i + 1])
                acc += s * part
            lanes[lane] = acc
        y[row] = np.float16(lanes.sum())                               # simd_sum
    return y

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for N, K in ((64, 256), (128, 1024), (96, 768)):
        W = rng.standard_normal((N, K)).astype(np.float32) * 0.05; x = rng.standard_normal(K).astype(np.float32)
        packed, scales, Wdq = pack_q4(W)
        ref = Wdq.astype(np.float32) @ x.astype(np.float16).astype(np.float32)
        y = kernel_emulation(packed, scales, x, K).astype(np.float32)
        fp16_bytes, q4_bytes = 2 * N * K, packed.nbytes + scales.nbytes
        print(f"N={N:4d} K={K:5d}  max rel err vs dequant ref = {np.abs(y - ref).max() / np.abs(ref).max():.2e}"
              f"  weight bytes fp16={fp16_bytes:,} q4={q4_bytes:,} ({fp16_bytes / q4_bytes:.2f}x smaller)")
