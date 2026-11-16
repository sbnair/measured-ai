"""A tiny ReLU position-sizing model used throughout the article.

Inputs  x = (signal, volatility, spread), each scaled to [0, 1].
Output  f(x) = target position as a fraction of the risk limit.
Weights are exact rationals, so NumPy, Z3 and Lean all see the same model.
"""
from fractions import Fraction as F
import numpy as np

# hidden layer: 4 ReLU units, 3 inputs (signal, volatility, spread)
W1 = [[F(2),    F(-1),    F(-1, 2)],
      [F(2),    F(-1, 2), F(0)],
      [F(1),    F(-1, 2), F(-1)],
      [F(1, 2), F(1, 2),  F(1, 2)]]
b1 = [F(0), F(-1, 2), F(1, 4), F(-1, 4)]

# output layer: one linear unit
W2 = [F(1), F(-1, 2), F(1, 2), F(-1, 4)]
b2 = F(0)

W1f, b1f = np.array(W1, dtype=float), np.array(b1, dtype=float)
W2f, b2f = np.array(W2, dtype=float), float(b2)

X0 = [F(3, 5), F(3, 10), F(1, 5)]   # today's inputs: signal 0.6, vol 0.3, spread 0.2
LIMIT = F(1)                         # position must never exceed 100% of the risk limit


def forward(x):
    """Float forward pass."""
    h = np.maximum(W1f @ np.asarray(x, dtype=float) + b1f, 0.0)
    return float(W2f @ h + b2f)
