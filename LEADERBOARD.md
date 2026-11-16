# Leaderboard

Submit a row with a pull request (see [CONTRIBUTING.md](CONTRIBUTING.md)). Every row links to the command that produces it.

## A. Language-model compression (03-edge-compression)

Model: 4-layer, width-256 byte-level GPT, 3.36M parameters, trained with `train.py target 256 8 4 2500`.
Metric: held-out bits per byte (lower is better), 256 windows of 128 bytes. Baseline: **1.8511**.

| Method | Bits / weight | Linear weights (MB) | Bits per byte | Δ vs baseline | Command | By |
|---|---|---|---|---|---|---|
| Full precision | 16 | 6.423 | 1.8511 | 0 | `run_quant.py` | @author |
| RTN int8, per tensor | 8 | 3.211 | 1.8510 | 0.000 | `run_quant.py` | @author |
| RTN int4, g32 | 4.5 | 1.806 | 1.8571 | +0.006 | `run_quant.py` | @author |
| GPTQ int4, g32 | 4.5 | 1.806 | 1.8553 | +0.004 | `run_quant.py` | @author |
| GPTQ int4 g32 + KV int8 | 4.5 | 1.806 | 1.8552 | +0.004 | `run_quant.py` | @author |
| RTN int3, g32 | 3.5 | 1.405 | 1.9034 | +0.052 | `run_quant.py` | @author |
| GPTQ int3, g32 | 3.5 | 1.405 | 1.8684 | +0.017 | `run_quant.py` | @author |
| GPTQ int2, g32 | 2.5 | 1.004 | 2.3630 | +0.512 | `run_quant.py` | @author |
| 2:4 + 400-step masked fine-tune | 16 (dense) | 6.423 | 1.9043 | +0.053 | `sparse_ft.py` | @author |

**Open target:** beat GPTQ int3 g32 (1.8684) at ≤ 3.5 bits per weight, or beat 1.9 bits/byte at ≤ 2.5 bits per weight.

## B. Vision RL agent under compression and shift (02-multimodal-rl + 03)

Agent: `pre_sft_sym`, 3 seeds. Metric: unsafe sends (% of 1,000 held-out episodes), lower is better.

| Weights | Clean | Shifted | Command | By |
|---|---|---|---|---|
| Full precision | 0.43% | 2.40% | `quant_agent.py` | @author |
| RTN 8-bit | 0.43% | 2.50% | `quant_agent.py` | @author |
| RTN 4-bit g32 | 0.73% | 2.90% | `quant_agent.py` | @author |
| RTN 3-bit g32 | 2.73% | 4.53% | `quant_agent.py` | @author |

**Open target:** a 3-bit agent at or below 1% unsafe sends on clean screens.

## C. Kernels on real hardware (03-edge-compression/metal)

| Kernel | Device | Shape (N × K) | Median time | vs fp16 GEMV | By |
|---|---|---|---|---|---|
| `gemv_q4` | *not yet measured* | | | | |

**Open target:** the first real timing on any Apple GPU.

(Replace `@author` with your GitHub handle before publishing.)
