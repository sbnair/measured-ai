"""Formal-verifier reward functions in the signature TRL's GRPOTrainer expects:
reward_fn(prompts, completions, **dataset_columns) -> list[float]
"""
import json
import os
import re
import subprocess
import tempfile

from z3 import Real, Solver, And, Or, sat

LEAN = os.environ.get("LEAN_BIN", "lean")
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}   # Lean's standard axioms


def _extract(text, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------- Lean 4 proofs
def check_lean_proof(statement, proof, timeout=60):
    """Return (reward, reason). Reward 1 only if Lean accepts the proof with standard axioms."""
    src = (f"theorem target : {statement} := by\n"
           + "\n".join("  " + line for line in proof.splitlines())
           + "\n\n#print axioms target\n")
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        out = subprocess.run([LEAN, path], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 0.0, "timeout"
    finally:
        os.unlink(path)
    log = out.stdout + out.stderr
    if out.returncode != 0 or "error" in log:
        return 0.0, "does not type-check"
    m = re.search(r"'target' depends on axioms: \[(.*?)\]", log, re.S)
    if m is None:
        if "'target' does not depend on any axioms" in log:
            return 1.0, "verified, no axioms"
        return 0.0, "axiom report missing"          # e.g. the proof smuggled in #exit
    used = {a.strip() for a in m.group(1).split(",")}
    if "sorryAx" in used:
        return 0.0, "uses sorry"
    if not used <= ALLOWED_AXIOMS:
        return 0.0, f"non-standard axioms {sorted(used - ALLOWED_AXIOMS)}"
    return 1.0, f"verified, axioms {sorted(used)}"


def lean_proof_reward(prompts, completions, statement, **kwargs):
    rewards = []
    for completion, stmt in zip(completions, statement):
        proof = _extract(completion, "proof")
        rewards.append(0.0 if proof is None else check_lean_proof(stmt, proof)[0])
    return rewards


# ------------------------------------------------------ Z3 risk-policy check
def check_sizing_rule(rule, max_pos=1.0):
    """Rule: position = a*signal - b*vol + c, for signal in [-1, 1] and vol in [0, 1].
    Policy: |position| <= max_pos everywhere, and higher volatility never raises the position."""
    a, b, c = float(rule["a"]), float(rule["b"]), float(rule["c"])
    s, v = Real("signal"), Real("vol")
    pos = a * s - b * v + c
    domain = And(s >= -1, s <= 1, v >= 0, v <= 1)
    solver = Solver()
    solver.add(domain, Or(pos > max_pos, pos < -max_pos))   # search for a breach
    if solver.check() == sat:
        m = solver.model()
        return 0.0, f"limit breached at signal={m[s]}, vol={m[v]}"
    if b < 0:
        return 0.0, "position rises with volatility"
    return 1.0, "proved within limit for all inputs"


def risk_policy_reward(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        try:
            rule = json.loads(_extract(completion, "rule") or "")
            rewards.append(check_sizing_rule(rule)[0])
        except (ValueError, KeyError, TypeError):
            rewards.append(0.0)                              # unparseable answer
    return rewards
