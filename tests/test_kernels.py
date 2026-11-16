"""Tiled attention and the Metal 4-bit GEMV emulation agree with their references."""
import numpy as np
from conftest import use
use("03-edge-compression"); use("03-edge-compression/metal")
import flash, emulate

def test_tiled_attention_is_exact():
    rng = np.random.default_rng(0); Q, K, V = (rng.standard_normal((300, 32)) for _ in range(3))
    O, peak = flash.tiled(Q, K, V)
    assert np.abs(O - flash.naive(Q, K, V)).max() < 1e-10 and peak <= 64 * 64

def test_metal_q4_gemv_emulation():
    rng = np.random.default_rng(1); W = (rng.standard_normal((32, 512)) * 0.05).astype(np.float32)
    x = rng.standard_normal(512).astype(np.float32); packed, scales, Wdq = emulate.pack_q4(W)
    ref = Wdq.astype(np.float32) @ x.astype(np.float16).astype(np.float32)
    y = emulate.kernel_emulation(packed, scales, x, 512).astype(np.float32)
    assert np.abs(y - ref).max() / np.abs(ref).max() < 2e-3
