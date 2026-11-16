"""Weight quantization (RTN and GPTQ), 2:4 pruning and KV-cache quantization for the small GPT."""
import copy, torch, torch.nn as nn

def fake_quant(W, bits, group):
    """Symmetric round-to-nearest. group=None: one scale per tensor; group=k: one scale per k inputs per row."""
    qmax = 2 ** (bits - 1) - 1
    out, inp = W.shape
    if group is None:
        s = W.abs().max() / qmax
        return (W / s).round().clamp(-qmax - 1, qmax) * s
    Wg = W.reshape(out, inp // group, group)
    s = (Wg.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-8)
    return ((Wg / s).round().clamp(-qmax - 1, qmax) * s).reshape(out, inp)

def linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]

def rtn(model, bits, group):
    m = copy.deepcopy(model)
    for _, lin in linears(m):
        lin.weight.data = fake_quant(lin.weight.data, bits, group)
    return m

@torch.no_grad()
def gptq(model, calib, bits, group, damp=0.01):
    """GPTQ (Frantar et al., 2022): quantize columns one at a time and push each column's
    rounding error onto the not-yet-quantized columns, weighted by the inverse Hessian H = 2 X^T X."""
    m = copy.deepcopy(model); qmax = 2 ** (bits - 1) - 1
    feats = {}
    hooks = [lin.register_forward_hook(lambda mod, i, o, n=n: feats.setdefault(n, []).append(i[0].reshape(-1, i[0].shape[-1])))
             for n, lin in linears(m)]
    m(calib)
    for h in hooks: h.remove()
    for n, lin in linears(m):
        X = torch.cat(feats[n]).double(); W = lin.weight.data.double().clone()
        H = 2 * X.T @ X / len(X)
        H += damp * H.diag().mean() * torch.eye(H.shape[0], dtype=H.dtype)
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
        Q = torch.zeros_like(W)
        for j in range(W.shape[1]):
            if j % group == 0:                                   # fresh scale for each group, from current weights
                s = (W[:, j:j + group].abs().amax(1) / qmax).clamp_min(1e-8)
            w = W[:, j]; q = (w / s).round().clamp(-qmax - 1, qmax) * s
            Q[:, j] = q
            err = (w - q) / Hinv[j, j]
            W[:, j:] -= err[:, None] * Hinv[j, j:][None, :]      # compensate the remaining columns
        lin.weight.data = Q.float()
    return m

def prune_2_4(model):
    """Semi-structured sparsity: in every group of 4 consecutive inputs keep the 2 largest weights."""
    m = copy.deepcopy(model)
    for _, lin in linears(m):
        W = lin.weight.data; o, i = W.shape
        g = W.reshape(o, i // 4, 4); idx = g.abs().topk(2, -1).indices
        mask = torch.zeros_like(g).scatter_(-1, idx, 1.0)
        lin.weight.data = (g * mask).reshape(o, i)
    return m

def kv_quantizer(bits):
    """Per-token, per-head symmetric quantization of cached keys/values (one scale per head-dim vector)."""
    qmax = 2 ** (bits - 1) - 1
    def q(t):
        s = (t.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-8)
        return (t / s).round().clamp(-qmax - 1, qmax) * s
    return q

def set_kv(model, bits):
    for b in model.blocks:
        b.kv_quant = None if bits is None else kv_quantizer(bits)

def weight_bytes(model, bits, group, scale_bits=16):
    """Storage for the Linear weights only (embeddings and norms stay 16-bit in every row)."""
    n = sum(l.weight.numel() for _, l in linears(model))
    scales = 0 if group is None else n // group
    return (n * bits + scales * scale_bits) / 8
