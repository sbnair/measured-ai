# 02 · Multi-modal agentic RL

The same task, but balance and limit arrive as 32×64 rendered screenshots (`screens.py`, needs the DejaVu fonts). A CNN reader feeds a decision MLP.

```bash
python mm_train.py scratch 0          # RL from scratch
python mm_train.py sft 0              # 10 demos + RL, no perception pre-training
python mm_train.py pre_sft 0          # pre-trained reader, soft features
python mm_train.py pre_sft_sym 0      # pre-trained reader + symbolic compare (main result)
python analyze.py pre_sft_sym 0       # clean vs shifted evaluation, guard sweep
python aggregate.py                   # all modes x 3 seeds (needs the 12 trained policies)
python quant_agent.py                 # compress the trained agent (uses ../03-edge-compression/quant.py)
```
