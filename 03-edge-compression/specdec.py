"""Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) with a small draft model and KV caches."""
import time, torch, torch.nn.functional as F

def new_caches(m): return [dict() for _ in m.blocks]

def trim(caches, n):
    """Roll a KV cache back to its first n positions (rejected draft tokens are discarded)."""
    for c in caches:
        if "k" in c: c["k"], c["v"] = c["k"][:, :, :n], c["v"][:, :, :n]

@torch.no_grad()
def autoregressive(target, prompt, n_new, temp=1.0, gen=None):
    caches = new_caches(target); x = prompt.clone(); calls = 0
    logits = target(x, caches); calls += 1
    for i in range(n_new):
        p = probs(logits[:, -1], temp); t = sample(p, temp, gen)
        x = torch.cat([x, t], 1)
        if i < n_new - 1:
            logits = target(t, caches, start=x.shape[1] - 1); calls += 1
    return x, calls

def probs(logits, temp):
    return F.softmax(logits / max(temp, 1e-6), -1)

def sample(p, temp, gen):
    return p.argmax(-1, keepdim=True) if temp == 0 else torch.multinomial(p, 1, generator=gen)

def clen(caches): return caches[0]["k"].shape[2] if "k" in caches[0] else 0

@torch.no_grad()
def speculative(target, draft, prompt, n_new, gamma=4, temp=1.0, gen=None):
    """Each round: the draft proposes gamma tokens; the target scores all of them in ONE forward pass;
    token i is kept with probability min(1, p_i/q_i); on the first rejection, resample from max(0, p - q);
    if all gamma are kept, the same pass also yields one bonus token. KV caches roll back on rejection."""
    tc, dc = new_caches(target), new_caches(draft); x = prompt.clone()
    target_calls = accepted = proposed = rejected = 0
    while x.shape[1] - prompt.shape[1] < n_new:
        n0 = x.shape[1]
        dl = draft(x[:, clen(dc):], dc, start=clen(dc))[:, -1]           # catch the draft up to x
        qs, toks = [], []
        for i in range(gamma):                                          # 1) draft proposes gamma tokens
            q = probs(dl, temp); t = sample(q, temp, gen); qs.append(q); toks.append(t)
            if i < gamma - 1: dl = draft(t, dc, start=n0 + i)[:, -1]
        prop = torch.cat(toks, 1); s0 = clen(tc)
        tlog = target(torch.cat([x[:, s0:], prop], 1), tc, start=s0)    # 2) ONE target pass verifies all
        target_calls += 1
        ps = [probs(tlog[:, n0 - 1 - s0 + i], temp) for i in range(gamma + 1)]
        n_ok = 0
        for i in range(gamma):                                          # 3) accept / reject left to right
            p, q, t = ps[i], qs[i], toks[i]
            ok = bool(p.argmax(-1) == t[0, 0]) if temp == 0 else \
                torch.rand(1, generator=gen).item() < min(1.0, (p[0, t] / q[0, t]).item())
            if not ok: break
            n_ok += 1
        proposed += gamma; accepted += n_ok; rejected += n_ok < gamma
        if n_ok < gamma:
            p, q = ps[n_ok], qs[n_ok]
            r = p if temp == 0 else (p - q).clamp_min(0)
            fix = sample(r / r.sum(), temp, gen)                        # corrected token
        else:
            fix = sample(ps[gamma], temp, gen)                          # bonus token
        x = torch.cat([x, prop[:, :n_ok], fix], 1)
        trim(tc, n0 + n_ok); trim(dc, min(clen(dc), n0 + n_ok))         # 4) roll back rejected positions
    # per-token acceptance rate: accepted / (tokens actually tested); later tokens after a rejection are never tested
    return x[:, :prompt.shape[1] + n_new], target_calls, accepted / max(accepted + rejected, 1)
