"""A small byte-level GPT, trained from scratch, used as the subject of every compression experiment."""
import math, torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def attn(self, x, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, -1)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        kvq = getattr(self, "kv_quant", None)
        if cache is not None:                       # KV cache: append this step's keys/values
            if "k" in cache:
                k = torch.cat([cache["k"], k], 2); v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        if kvq is not None:                         # what attention reads back from a quantized KV cache
            k, v = kvq(k), kvq(v)
        causal = cache is None or T > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal and k.shape[2] == T)
        if cache is not None and T > 1 and k.shape[2] != T:   # multi-token step on top of a cache
            S = k.shape[2]; mask = torch.ones(T, S, dtype=torch.bool).tril(S - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))

class GPT(nn.Module):
    def __init__(self, d=256, h=8, L=4, ctx=256, vocab=256):
        super().__init__()
        self.ctx = ctx
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, idx, caches=None, start=0):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(start, start + T))
        for i, b in enumerate(self.blocks):
            x = b(x, None if caches is None else caches[i])
        return self.head(self.ln(x))

def batches(data, B, T, gen):
    i = torch.randint(len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([data[j:j + T] for j in i]); y = torch.stack([data[j + 1:j + T + 1] for j in i])
    return x, y

def load_data():
    raw = torch.tensor(list(open("corpus.bin", "rb").read()), dtype=torch.long)
    n = int(0.95 * len(raw))
    return raw[:n], raw[n:]                       # train, held-out

@torch.no_grad()
def eval_bpb(model, val, T=128, n=64, seed=0):
    """Bits per byte on held-out text (lower is better)."""
    g = torch.Generator().manual_seed(seed); model.eval(); tot = 0.0
    for _ in range(n // 16):
        x, y = batches(val, 16, T, g)
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    return tot / (n // 16) / math.log(2)
