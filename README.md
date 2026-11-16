# measured-ai

**Small, reproducible experiments on the hard parts of modern AI: agentic RL, multi-modal agents, on-device compression and formal verification. Every number comes from code in this repo, and everything not measured is labelled as not measured.**

Hello everyone. I am Shailesh, and I have spent more than 23 years building HFT engines, FPGA pipelines and trading risk systems, where you never trust a number you did not measure yourself. This repo is where I bring that habit to AI. Each folder is one experiment from one of my articles, small enough to rerun on a laptop CPU, with its results checked in.

Contributions are very welcome, especially results on hardware I do not have. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [open challenges](docs/first-issues.md).

## Headline results

| Experiment | What was measured | Result |
|---|---|---|
| [01-agentic-rl](01-agentic-rl) | GRPO on a tool-using payment agent with a spec verifier | 10 demonstrations + GRPO: **100% / 100% / 99.95%** spec compliance across 3 seeds; from scratch it stalls at ~36% |
| [01-agentic-rl](01-agentic-rl) | Reward hacking | An outcome-only reward taught the agent to look up the compliance limit **and then ignore it** in 13.4% of episodes |
| [02-multimodal-rl](02-multimodal-rl) | Agent reads balance and limit from screenshots | **98.4%** spec compliance with a pre-trained reader and a symbolic comparison; unsafe sends rise ~5x on shifted screens |
| [02-multimodal-rl](02-multimodal-rl) | Runtime re-capture guard | Unsafe sends on shifted screens **2.02% → 0.03%**, at 6.6% escalation to a human |
| [03-edge-compression](03-edge-compression) | 4-bit weights + 8-bit KV cache on a 3.4M-param GPT | Linear weights **3.6x smaller** for **+0.004** bits/byte |
| [03-edge-compression](03-edge-compression) | 3-bit: GPTQ vs round-to-nearest | GPTQ cuts the quality loss by **two-thirds** with 4 KB of calibration text |
| [03-edge-compression](03-edge-compression) | Speculative decoding | **3.5x fewer** target passes at γ=4, greedy output token-identical |
| [03-edge-compression](03-edge-compression) | Compressing the vision agent, in and out of distribution | 3-bit on shifted screens: **~10x** the unsafe sends of full precision on clean screens |
| [03-edge-compression/metal](03-edge-compression/metal) | 4-bit GEMV kernel in Metal | Logic validated lane by lane in emulation. **Not yet compiled or timed on Apple GPU** ([help wanted](docs/first-issues.md)) |
| [04-formal-verification](04-formal-verification) | LLM-written SVA, AI-generated RTL, formal model-risk checks | Arbiter proved by k-induction, 7/9 mutants killed; AI FIFO proved with PDR, optimised rewrite proved equivalent |

Raw outputs are in [results/](results).

## Repository layout

```
01-agentic-rl/            env.py (task + spec verifier), train.py (GRPO, SFT warm start)
02-multimodal-rl/         screens.py (rendered screenshots), mm_train.py, analyze.py, guard.py, quant_agent.py
03-edge-compression/      model.py, quant.py (RTN, GPTQ, 2:4, KV), specdec.py, flash.py, metal/
04-formal-verification/   ai-model-risk/ (IBP, Z3, Lean 4), post-training/ (verifiable rewards),
                          llm-sva/ (SymbiYosys), rtl-loop/ (LLM RTL + PDR + equivalence)
results/                  raw JSON/JSONL from the runs reported in the articles
tests/                    fast correctness tests, run in CI on every push
```

## Quick start

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
python -m pytest -q tests                      # ~30 s, no trained weights needed

# main results (CPU)
cd 01-agentic-rl && python train.py verified_v2 sft=10          # ~4 min
cd 02-multimodal-rl && python mm_train.py pre_sft_sym 0         # ~2 min
cd 03-edge-compression && python make_corpus.py && python train.py target 256 8 4 2500 && python run_quant.py
```

The formal-verification folders need the free [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) (Yosys, SymbiYosys, solvers) and, for the Lean file, [Lean 4](https://lean-lang.org/).

## Articles

Each folder goes with one article. Links are added as they are published:

- *Agentic RL from Scratch: Training a Tool-Using Agent with GRPO and Verifiable Rewards* (link: TODO)
- *Agentic RL with Eyes: Training a Payment Agent That Reads Its Own Screenshots* (link: TODO)
- *Shrinking a Transformer for the Edge: Quantization, Pruning, KV Cache and Speculative Decoding, Measured* (link: TODO)
- *Proving, Not Just Testing: Formal Verification for AI Model Risk* (link: TODO)
- *Using LLMs to write SVA properties* and *Closing the Neurosymbolic Loop: Formally Verifying AI-Generated RTL* (links: TODO)

## Ground rules

1. Every reported number has a script and a seed that produces it.
2. Anything not measured is labelled as not measured.
3. Small models are a feature: they keep every claim checkable on a laptop. Results on larger models and real hardware are the most valuable contributions.

## Licence

MIT. See [LICENSE](LICENSE).
