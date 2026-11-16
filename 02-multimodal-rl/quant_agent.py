"""Compress the vision-based RL payment agent (from the multi-modal article) and test it in and out of distribution."""
import copy, json, sys, torch, torch.nn as nn
import mm_train as M, analyze as A
from screens import bank
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "03-edge-compression")); import quant as Q

def quantize_agent(p, bits, group=32):
    q = copy.deepcopy(p)
    for mod in q.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)) and bits is not None:
            W = mod.weight.data; flat = W.reshape(W.shape[0], -1)
            g = group if flat.shape[1] % group == 0 else flat.shape[1]
            mod.weight.data = Q.fake_quant(flat, bits, g).reshape(W.shape)
    return q

clean, shifted = bank(10, 999), bank(10, 777, noise=0.16, blur=0.6)
for s in (0, 1, 2):
    p = M.Policy(symbolic=True); p.load_state_dict(torch.load(f"policy_pre_sft_sym_s{s}.pt")); p.eval()
    for bits in (None, 8, 4, 3, 2):
        q = quantize_agent(p, bits)
        for name, imgs in (("clean", clean), ("shifted", shifted)):
            eps = A.run(q, imgs, n=1000)
            r = A.summary(eps)
            print(json.dumps({"seed": s, "bits": bits or 32, "cond": name, "reader_acc": M.reader_accuracy(q.reader, imgs),
                              "correct": r["correct"], "unsafe": r["violation"]}), flush=True)
