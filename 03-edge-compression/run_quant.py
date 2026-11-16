"""Every compression experiment in the article, on the trained target model. Prints one JSON line per config."""
import json, torch
from model import GPT, load_data, eval_bpb, batches
import quant as Q
torch.set_num_threads(2); torch.manual_seed(0)
tr, va = load_data()
m = GPT(256, 8, 4); m.load_state_dict(torch.load("target.pt")); m.eval()
calib, _ = batches(tr, 32, 128, torch.Generator().manual_seed(1))   # 4,096 bytes of calibration text
N = 256                                                              # 256 held-out windows x 128 bytes
lin_params = sum(l.weight.numel() for _, l in Q.linears(m))
def report(name, model, bits, group, extra=None):
    row = {"config": name, "bpb": round(eval_bpb(model, va, n=N), 4),
           "weight_MB": round(Q.weight_bytes(m, bits, group) / 1e6, 3),
           "bits_per_weight": round(bits + (0 if group is None else 16 / group), 3)}
    if extra: row.update(extra)
    print(json.dumps(row), flush=True)
print(json.dumps({"linear_params": lin_params, "total_params": sum(p.numel() for p in m.parameters())}))
report("fp16 baseline", m, 16, None)
for bits, group, label in [(8, None, "per-tensor"), (8, 256, "per-channel"), (4, None, "per-tensor"), (4, 256, "per-channel"),
                           (4, 128, "g128"), (4, 64, "g64"), (4, 32, "g32"), (3, 256, "per-channel"), (3, 32, "g32"), (2, 32, "g32")]:
    g = None if group is None else group
    # per-channel = one scale per row: use group = the layer's input width
    if label == "per-channel":
        mq = __import__("copy").deepcopy(m)
        for _, lin in Q.linears(mq): lin.weight.data = Q.fake_quant(lin.weight.data, bits, lin.weight.shape[1])
        report(f"RTN int{bits} {label}", mq, bits, 256)          # bytes approximated with one scale per 256 weights
    else:
        report(f"RTN int{bits} {label}", Q.rtn(m, bits, g), bits, g)
for bits, group in [(4, 32), (3, 32), (2, 32)]:
    report(f"GPTQ int{bits} g{group}", Q.gptq(m, calib, bits, group), bits, group)
p = Q.prune_2_4(m)
report("2:4 sparse (magnitude)", p, 16, None, {"note": "half the weights are zero; bytes shown are dense fp16"})
report("2:4 sparse + RTN int4 g32", Q.rtn(p, 4, 32), 4, 32, {"note": "dense storage of the sparse weights"})
for kvb in (8, 4, 3):
    Q.set_kv(m, kvb); report(f"KV cache int{kvb} (weights fp16)", m, 16, None)
Q.set_kv(m, None)
mq = Q.gptq(m, calib, 4, 32); Q.set_kv(mq, 8)
report("GPTQ int4 g32 + KV int8", mq, 4, 32)
