import sys, time, torch, torch.nn.functional as F
from model import GPT, batches, load_data, eval_bpb
torch.set_num_threads(1)
name, d, h, L, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
torch.manual_seed(0); g = torch.Generator().manual_seed(0)
tr, va = load_data(); m = GPT(d, h, L)
print(name, "params", sum(p.numel() for p in m.parameters()), flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1); t0 = time.time()
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
for s in range(steps + 1):
    if s % 250 == 0:
        print(s, "val bpb", round(eval_bpb(m, va), 4), "sec", round(time.time() - t0), flush=True); m.train()
    if s == steps: break
    x, y = batches(tr, 16, 128, g)
    loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
torch.save(m.state_dict(), f"{name}.pt")
