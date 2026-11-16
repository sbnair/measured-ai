"""Exact verification of the position-sizing model with the Z3 SMT solver.

We ask Z3 for a counterexample. "unsat" means none exists, so the
property is proved for every input in the region. "sat" returns a
concrete input that breaks it.
"""
from fractions import Fraction as F
from z3 import Real, Solver, Optimize, If, And, unsat
from model import W1, b1, W2, b2, X0, LIMIT


def relu(z):
    return If(z > 0, z, 0)


def net(x):
    h = [relu(sum(W1[j][i] * x[i] for i in range(3)) + b1[j]) for j in range(4)]
    return sum(W2[j] * h[j] for j in range(4)) + b2


def in_box(x, centre, eps):
    return And(*[And(x[i] >= centre[i] - eps, x[i] <= centre[i] + eps) for i in range(3)])


def check_limit(eps):
    x = [Real(n) for n in ("signal", "vol", "spread")]
    s = Solver()
    s.add(in_box(x, X0, eps), net(x) > LIMIT)          # look for a breach
    if s.check() == unsat:
        return "PROVED: f(x) <= limit for every x in the box", None
    m = s.model()
    cex = [m.eval(v).as_fraction() for v in x]
    return "COUNTEREXAMPLE", cex


def exact_max(eps):
    x = [Real(n) for n in ("signal", "vol", "spread")]
    o = Optimize()
    o.add(in_box(x, X0, eps))
    h = o.maximize(net(x))
    o.check()
    return o.upper(h).as_fraction()


def absval(e):
    return If(e >= 0, e, -e)


def check_monotone_in_vol(size=absval):
    """For all x in [0,1]^3: raising volatility must not raise the position size."""
    a = [Real(f"a_{n}") for n in ("signal", "vol", "spread")]
    b = [Real(f"b_{n}") for n in ("signal", "vol", "spread")]
    s = Solver()
    for v in a + b:
        s.add(v >= 0, v <= 1)
    s.add(b[0] == a[0], b[2] == a[2], b[1] > a[1])      # only volatility rises
    s.add(size(net(b)) > size(net(a)))                  # ...and the position grows
    if s.check() == unsat:
        return "PROVED: position never grows when volatility rises", None
    m = s.model()
    return "COUNTEREXAMPLE", ([m.eval(v).as_fraction() for v in a],
                              [m.eval(v).as_fraction() for v in b])


def f_exact(x):
    h = [max(sum(W1[j][i] * x[i] for i in range(3)) + b1[j], 0) for j in range(4)]
    return sum(W2[j] * h[j] for j in range(4)) + b2


if __name__ == "__main__":
    for eps in [F(1, 20), F(2, 25), F(1, 10)]:
        verdict, cex = check_limit(eps)
        print(f"eps = {float(eps):.2f}  exact max = {float(exact_max(eps)):.4f}  {verdict}")
        if cex:
            print("   x =", [float(c) for c in cex], " f(x) =", float(f_exact(cex)))
    verdict, cex = check_monotone_in_vol(size=absval)   # |position| must not grow
    print("monotonicity of |f| in volatility:", verdict)
    if cex:
        a, b = cex
        print("   a =", [float(v) for v in a], " f(a) =", float(f_exact(a)))
        print("   b =", [float(v) for v in b], " f(b) =", float(f_exact(b)))
