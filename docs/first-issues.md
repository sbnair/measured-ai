# First open challenges

Copy each of these into a GitHub issue with the label shown. They are ordered from easiest to hardest.

1. **[good first issue] Reproduce the agentic-RL headline on your machine.** Run `01-agentic-rl/train.py verified_v2 sft=10` for seeds 0–2 and post the final spec-compliance numbers and your CPU. Reproductions on different machines are valuable on their own.

2. **[good first issue] Add a `--quick` mode to every training script.** A setting that finishes in under a minute, so the full pipeline can run in CI.

3. **[hardware] Compile and time `gemv_q4.metal` on an Apple GPU.** Write a small Swift or Objective-C host program, check its output against `metal/emulate.py`, and report the median time against an fp16 GEMV of the same shape. This fills leaderboard C.

4. **[research] GPTQ for the vision agent's CNN reader.** The agent loses most of its accuracy at 3 bits with round-to-nearest. Apply `quant.gptq`-style error compensation to the conv and linear layers of the reader and report leaderboard B.

5. **[research] Freeze the reader during RL.** In the multi-modal runs, RL fine-tuning lowered reading accuracy in all six warm-started runs. Add a frozen-reader option and compare spec compliance and unsafe sends across 3 seeds.

6. **[research] EAGLE-style draft head.** Replace the separate draft model in `specdec.py` with a small head that predicts the target's second-to-top-layer features, and report acceptance rate and tokens per target pass against the current draft model.

7. **[research] SparseGPT for 2:4.** One-shot magnitude 2:4 pruning costs +0.55 bits per byte. Implement SparseGPT-style error compensation and see how much of the gap closes without fine-tuning.

8. **[hardware] Control-loop latency benchmark.** Time a forward pass of the vision agent at full precision, int8 and int4 on a Raspberry Pi, a Jetson or a phone, and report the highest control frequency each meets with p99 latency.

9. **[formal] Prove the runtime guard.** Write a Lean 4 or Z3 proof that the re-capture guard in `02-multimodal-rl/guard.py` never lets a SEND through when every re-read is correct, and state precisely what it assumes about the reader.

10. **[research] Larger model, same pipeline.** Run `run_quant.py` and `run_spec.py` on a 50–150M-parameter open model, and report where the small-model conclusions hold and where they break. This is the most important open question in the repo.
