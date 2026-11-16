import json, torch, statistics as st
import analyze as A, mm_train as M, guard as Gd
from screens import bank
clean, shifted = bank(10, 999), bank(10, 777, noise=0.16, blur=0.6)
res = {}
for mode in ("scratch", "sft", "pre_sft", "pre_sft_sym"):
    for s in (0, 1, 2):
        p = M.Policy(symbolic="sym" in mode); p.load_state_dict(torch.load(f"policy_{mode}_s{s}.pt")); p.eval()
        row = {}
        for name, imgs in (("clean", clean), ("shifted", shifted)):
            eps = A.run(p, imgs)
            row[name] = {"reader_acc": M.reader_accuracy(p.reader, imgs), "no_guard": A.summary(eps),
                         "by_margin": A.by_margin(eps), "guard_k3": Gd.summary_with(p, imgs, eps, k=3, tau=0.0)}
        res[f"{mode}_s{s}"] = row
        print(mode, s, {n: (row[n]["reader_acc"], row[n]["no_guard"]["correct"], row[n]["no_guard"]["violation"], row[n]["guard_k3"]["violation"], row[n]["guard_k3"]["escalated"]) for n in row}, flush=True)
json.dump(res, open("aggregate.json", "w"), indent=1)
