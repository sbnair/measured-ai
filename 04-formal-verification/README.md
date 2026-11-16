# 04 · Formal verification for AI

| Folder | What it shows | Needs |
|---|---|---|
| `ai-model-risk/` | Interval bound propagation, Z3 exact checks and a Lean 4 runtime guard for a position-sizing network | Python, z3-solver, Lean 4 |
| `post-training/` | Verifiable rewards (Lean proof check, Z3 rule check), DPO loss, GRPO advantages | Python, z3-solver, Lean 4; `trl` only for `check_configs.py` |
| `llm-sva/` | LLM-drafted SVA for a round-robin arbiter, proved with SymbiYosys; mutation testing | OSS CAD Suite |
| `rtl-loop/` | LLM-written FIFO proved with PDR; counterexample-to-prompt loop; equivalence of an optimised rewrite | OSS CAD Suite |

```bash
cd llm-sva  && sby -f prove.sby && python mutate.py
cd rtl-loop && sby -f fifo.sby && python rtl_loop.py fifo_bad.v && sby -f equiv.sby
```
