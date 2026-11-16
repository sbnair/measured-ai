import torch, json
from model import GPT
import quant as Q
m = GPT(256, 8, 4); m.load_state_dict(torch.load("target.pt"))
for n, l in Q.linears(m):
    W = l.weight.data
    print(json.dumps({"layer": n, "shape": list(W.shape), "max_over_std": round((W.abs().max() / W.std()).item(), 1),
                      "row_max_spread": round((W.abs().amax(1).max() / W.abs().amax(1).median()).item(), 2)}))
