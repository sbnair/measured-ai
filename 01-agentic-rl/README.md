# 01 · Agentic RL from scratch

A payment agent must call `CHECK_BALANCE` / `CHECK_LIMIT` before it can `SEND` or `REFUSE`. The reward is a verifier built from a four-clause spec (S1–S4 in `env.py`). Training is GRPO with an optional supervised warm start.

```bash
python train.py naive                 # outcome-only reward: learns to ignore the limit
python train.py verified_v1           # verifier without S4: learns to refuse everything
python train.py verified_v2           # full verifier from scratch: stalls ~36%
python train.py verified_v2 sft=10    # 10 demonstrations + GRPO: ~100% spec compliance
python train.py verified_v2 sft=10 seed=1
```

Options: `ent_coef=`, `lr=`, `G=`, `iters=`, `sft=`, `seed=`. Each run is a few minutes on one CPU core. Logs land in `log_*.json`.
