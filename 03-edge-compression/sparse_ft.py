"""2:4 magnitude pruning followed by a short fine-tune with the sparsity mask held fixed."""
import json, torch, torch.nn.functional as F
from model import GPT, batches, load_data, eval_bpb
import quant as Q
torch.set_num_threads(1); torch.manual_seed(0)
tr, va = load_data(); m = GPT(256, 8, 4); m.load_state_dict(torch.load("target.pt"))
p = Q.prune_2_4(m); masks = {n: (l.weight != 0).float() for n, l in Q.linears(p)}
print(json.dumps({"steps": 0, "bpb": round(eval_bpb(p, va, n=256), 4)}), flush=True)
opt = torch.optim.AdamW(p.parameters(), lr=3e-4, weight_decay=0.0); g = torch.Generator().manual_seed(5)
for s in range(1, 401):
    p.train(); x, y = batches(tr, 16, 128, g)
    loss = F.cross_entropy(p(x).flatten(0, 1), y.flatten()); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        for n, l in Q.linears(p): l.weight.mul_(masks[n])          # keep the 2:4 pattern
    if s % 100 == 0:
        print(json.dumps({"steps": s, "bpb": round(eval_bpb(p, va, n=256), 4)}), flush=True)
pq = Q.rtn(p, 4, 32); print(json.dumps({"steps": 400, "plus_int4_g32_bpb": round(eval_bpb(pq, va, n=256), 4)}))
# sanity: still 2:4
W = Q.linears(p)[0][1].weight; print("zeros fraction", round((W == 0).float().mean().item(), 3))
