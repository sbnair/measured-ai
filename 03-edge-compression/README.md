# 03 · Edge compression, measured

```bash
python make_corpus.py                       # Python stdlib source -> corpus.bin
python train.py target 256 8 4 2500         # 3.36M-param target, ~25 min on one core
python train.py draft 128 4 1 4000          # 0.30M-param draft, ~4 min
python run_quant.py                         # RTN, GPTQ, 2:4, KV-cache quantization
python sparse_ft.py                         # 2:4 + masked fine-tune
python run_spec.py                          # speculative decoding
python flash.py                             # tiled online-softmax attention check
python metal/emulate.py                     # Metal q4 GEMV logic check (not a GPU timing)
```

`metal/gemv_q4.metal` has **not** been compiled or timed on an Apple GPU yet. That is open challenge #3.
