"""Score candidate completions with the two verifier rewards."""
import time
from rewards import check_lean_proof, check_sizing_rule, lean_proof_reward

stmt = "∀ a b : Nat, a + b = b + a"
lean_cases = [
    ("correct proof",            stmt, "intro a b\nexact Nat.add_comm a b"),
    ("wrong proof",              stmt, "intro a b\nrfl"),
    ("gives up with sorry",      stmt, "sorry"),
    ("hides sorry behind #exit", stmt, "sorry\n#exit"),
    ("decide (kernel-checked)",  "2 ^ 10 = 1024", "decide"),
    ("native_decide (trusts the compiler)", "2 ^ 10 = 1024", "native_decide"),
]
print("Lean 4 proof reward")
for name, s, proof in lean_cases:
    t = time.time()
    reward, reason = check_lean_proof(s, proof)
    print(f"  {name:<38} reward={reward:.0f}  {reason}  ({time.time() - t:.1f}s)")

# The same check through the TRL-style interface:
completions = ["<proof>intro a b\nexact Nat.add_comm a b</proof>", "no proof here"]
print("  batch via lean_proof_reward:", lean_proof_reward(["p", "p"], completions, statement=[stmt, stmt]))

print("\nZ3 risk-policy reward")
rules = [
    ("conservative rule", {"a": 0.6, "b": 0.3, "c": 0.0}),
    ("too aggressive",    {"a": 0.9, "b": 0.0, "c": 0.2}),
    ("sizes up in vol",   {"a": 0.5, "b": -0.2, "c": 0.0}),
]
for name, rule in rules:
    reward, reason = check_sizing_rule(rule)
    print(f"  {name:<18} {rule}  reward={reward:.0f}  {reason}")
