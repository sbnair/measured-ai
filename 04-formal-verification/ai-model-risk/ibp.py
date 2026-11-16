"""Interval Bound Propagation (IBP) for the position-sizing model.

For an affine layer z = W x + b with x in [l, u]:
    centre c = (u + l) / 2, radius r = (u - l) / 2
    z lies in [W c + b - |W| r,  W c + b + |W| r]
ReLU is monotone, so it maps [l, u] to [max(l, 0), max(u, 0)].
"""
import numpy as np
from model import W1f, b1f, W2f, b2f, X0, LIMIT, forward


def affine(W, b, l, u):
    c, r = (u + l) / 2, (u - l) / 2
    mid, rad = W @ c + b, np.abs(W) @ r
    return mid - rad, mid + rad


def ibp(x0, eps):
    x0 = np.asarray(x0, dtype=float)
    l, u = affine(W1f, b1f, x0 - eps, x0 + eps)
    l, u = np.maximum(l, 0), np.maximum(u, 0)          # ReLU
    l, u = affine(W2f[None, :], np.array([b2f]), l, u)
    return float(l[0]), float(u[0])


def sampled_max(x0, eps, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.asarray(x0, dtype=float) + rng.uniform(-eps, eps, size=(n, 3))
    h = np.maximum(xs @ W1f.T + b1f, 0)
    return float((h @ W2f + b2f).max())


if __name__ == "__main__":
    print(f"f(x0) = {forward(X0):.4f}   limit = {float(LIMIT)}")
    print(f"{'eps':>5} {'IBP lower':>10} {'IBP upper':>10} {'sampled max':>12}  IBP verdict")
    for eps in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
        lo, hi = ibp(X0, eps)
        verdict = "proved safe" if hi <= LIMIT else "inconclusive"
        print(f"{eps:5.2f} {lo:10.4f} {hi:10.4f} {sampled_max(X0, eps):12.4f}  {verdict}")
