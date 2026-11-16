"""Speculative decoding on CPU: acceptance rate, tokens per target pass, wall-clock speedup, exactness."""
import json, time, torch
from model import GPT, load_data, eval_bpb
import specdec as S
torch.set_num_threads(1)
tr, va = load_data()
t = GPT(256, 8, 4); t.load_state_dict(torch.load("target.pt")); t.eval()
d = GPT(128, 4, 1); d.load_state_dict(torch.load("draft.pt")); d.eval()
print(json.dumps({"target_bpb": round(eval_bpb(t, va, n=256), 4), "draft_bpb": round(eval_bpb(d, va, n=256), 4),
                  "target_params": sum(p.numel() for p in t.parameters()), "draft_params": sum(p.numel() for p in d.parameters())}))
starts = [1000, 40000, 90000, 150000, 200000]; prompts = [va[s:s + 32][None] for s in starts]; NEW = 96
for temp in (0.0, 1.0):
    gen = torch.Generator().manual_seed(0); t0 = time.time(); base_calls = 0; base = []
    for p in prompts:
        x, c = S.autoregressive(t, p, NEW, temp, gen); base.append(x); base_calls += c
    t_base = time.time() - t0
    for gamma in (1, 2, 3, 4, 6, 8):
        gen = torch.Generator().manual_seed(0); t0 = time.time(); calls = 0; accs = []; same = True
        for p, b in zip(prompts, base):
            x, c, a = S.speculative(t, d, p, NEW, gamma, temp, gen); calls += c; accs.append(a)
            if temp == 0: same &= torch.equal(x, b)
        el = time.time() - t0; alpha = sum(accs) / len(accs)
        print(json.dumps({"temp": temp, "gamma": gamma, "alpha": round(alpha, 3),
                          "tokens_per_target_pass": round(len(prompts) * NEW / calls, 3),
                          "predicted_tokens_per_pass": round((1 - alpha ** (gamma + 1)) / (1 - alpha), 3),
                          "target_passes": calls, "baseline_passes": base_calls,
                          "wall_speedup_cpu": round(t_base / el, 3), "greedy_identical": same if temp == 0 else None}), flush=True)
