"""GRPO's group-relative advantage for binary verifier rewards."""
import numpy as np


def group_advantages(rewards, eps=1e-4):
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)


groups = {
    "2 of 8 proofs verified": [1, 0, 0, 1, 0, 0, 0, 0],
    "7 of 8 proofs verified": [1, 1, 1, 1, 1, 1, 0, 1],
    "0 of 8 proofs verified": [0, 0, 0, 0, 0, 0, 0, 0],
    "8 of 8 proofs verified": [1, 1, 1, 1, 1, 1, 1, 1],
}
for name, rewards in groups.items():
    adv = group_advantages(rewards)
    print(f"{name}:  advantage(correct) = {adv.max():+.3f}   "
          f"advantage(wrong) = {adv.min():+.3f}")
