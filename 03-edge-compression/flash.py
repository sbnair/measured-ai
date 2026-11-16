"""FlashAttention's core idea in NumPy: tiled attention with an online softmax, never forming the N x N matrix."""
import numpy as np

def naive(Q, K, V):
    S = Q @ K.T / np.sqrt(Q.shape[1])                 # N x N scores: O(N^2) memory
    P = np.exp(S - S.max(1, keepdims=True))
    return (P / P.sum(1, keepdims=True)) @ V

def tiled(Q, K, V, Br=64, Bc=64):
    N, d = Q.shape; O = np.zeros_like(Q); scale = 1 / np.sqrt(d); peak = 0
    for i in range(0, N, Br):                          # one query tile at a time ("threadgroup")
        q = Q[i:i + Br]; m = np.full(len(q), -np.inf); l = np.zeros(len(q)); acc = np.zeros((len(q), d))
        for j in range(0, N, Bc):                      # stream key/value tiles through fast memory
            s = q @ K[j:j + Bc].T * scale              # Br x Bc block only
            peak = max(peak, s.size)
            m_new = np.maximum(m, s.max(1))
            corr = np.exp(m - m_new)                   # rescale what was accumulated so far
            p = np.exp(s - m_new[:, None])
            l = l * corr + p.sum(1)
            acc = acc * corr[:, None] + p @ V[j:j + Bc]
            m = m_new
        O[i:i + Br] = acc / l[:, None]
    return O, peak

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for N in (256, 1024, 4096):
        Q, K, V = (rng.standard_normal((N, 64)) for _ in range(3))
        O, peak = tiled(Q, K, V)
        err = np.abs(O - naive(Q, K, V)).max()
        print(f"N={N:5d}  max|tiled-naive|={err:.2e}  score elements held: tiled={peak:,}  naive={N*N:,}")
