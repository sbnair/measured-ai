"""Multi-modal agentic RL: the balance and limit arrive as images; the policy must read them."""
import json, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from env import PaymentAgentEnv, verified_reward, check_spec, HI, MAX_STEPS, SEND, REFUSE
from screens import bank

torch.set_num_threads(1)
BLANK = torch.zeros(1, 1, 32, 64)

class Reader(nn.Module):
    """Perception: screenshot -> distribution over the 20 possible values."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU())
        self.head = nn.Linear(32 * 4 * 8, HI)
    def forward(self, x):                      # x: [n,1,32,64]
        return self.head(self.conv(1 - x).flatten(1))   # logits over values 1..HI

class Policy(nn.Module):
    """Decision: reads each revealed screen into (expected value, confidence), then picks a tool."""
    def __init__(self, symbolic=False):
        super().__init__()
        self.reader = Reader(); self.symbolic = symbolic
        self.vals = torch.arange(1, HI + 1, dtype=torch.float32) / HI
        self.mlp = nn.Sequential(nn.Linear(10, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4))
    def read(self, img):
        p = F.softmax(self.reader(img), -1)
        return (p * self.vals).sum(-1), p.max(-1).values          # expected value / HI, confidence
    def forward(self, o):
        a = o["amount"] / HI
        feats = [torch.tensor([a, o["steps"] / MAX_STEPS])]
        for key in ("balance_img", "limit_img"):
            img = o[key]
            if img is None:
                feats.append(torch.zeros(4))
            else:
                v, c = self.read(img)
                if self.symbolic:      # read a symbol, then compare in code: amount <= read value
                    sym = self.reader(img).argmax(-1).float() + 1
                    feats.append(torch.cat([torch.ones(1), v, c, (o["amount"] <= sym).float()]))
                else:
                    feats.append(torch.cat([torch.ones(1), v, c, a - v]))   # seen, value, confidence, amount - value
        return torch.distributions.Categorical(logits=self.mlp(torch.cat(feats)))

class ScreenEnv(PaymentAgentEnv):
    """Same task and spec as before; tool results are screenshots, not numbers."""
    def __init__(self, images, rng):
        super().__init__(rng); self.images = images
    def obs(self):
        o = super().obs()
        pick = lambda kind, v: self.images[kind][v][self.rng.randrange(len(self.images[kind][v]))][None]
        o["balance_img"] = pick("BAL", self.balance) if self.seen_balance else None
        o["limit_img"] = pick("LIM", self.limit) if self.seen_limit else None
        return o

def rollout(policy, task, images, rng):
    env = ScreenEnv(images, rng); o = env.reset(task); logps = []; done = False
    while not done:
        dist = policy(o); a = dist.sample(); logps.append(dist.log_prob(a)); o, done = env.step(a.item())
    return env, torch.stack(logps).sum(), verified_reward(env)

def evaluate(policy, images, n=1000, seed=123, guard=None):
    rng = random.Random(seed); s = dict(correct=0, violation=0, unjustified_refuse=0, escalated=0, tools=0)
    with torch.no_grad():
        for _ in range(n):
            task = tuple(rng.randint(1, HI) for _ in range(3))
            env, _, _ = rollout(policy, task, images, rng)
            if guard and env.trace[-1] == SEND and guard(policy, env):
                s["escalated"] += 1; continue          # blocked before execution, sent to a human
            v = check_spec(env)
            s["correct"] += not v; s["violation"] += ("S1" in v or "S2" in v)
            s["unjustified_refuse"] += ("S3" in v or "S4" in v)
            s["tools"] += sum(a < 2 for a in env.trace)
    return {k: round(x / n, 4) for k, x in s.items()}

def expert_action(o, env):
    if o["balance_img"] is None: return 0
    if env.amount > env.balance: return 3
    if o["limit_img"] is None: return 1
    return 3 if env.amount > env.limit else 2

def sft(policy, images, n_demos, epochs=60, seed=0, freeze_reader=False):
    rng = random.Random(seed); data = []
    for _ in range(n_demos):
        env = ScreenEnv(images, rng); o = env.reset(tuple(rng.randint(1, HI) for _ in range(3))); done = False
        while not done:
            a = expert_action(o, env); data.append((o, a)); o, done = env.step(a)
    params = policy.mlp.parameters() if freeze_reader else policy.parameters()
    opt = torch.optim.Adam(params, lr=3e-3)              # 27 pairs would wreck a pretrained reader
    for _ in range(epochs):
        loss = -torch.stack([policy(o).log_prob(torch.tensor(a)) for o, a in data]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return len(data)

def pretrain_reader(reader, images, steps=2000, seed=0):
    """Supervised perception: labelled screenshots -> value. No agent, no reward."""
    g = torch.Generator().manual_seed(seed)
    X = torch.cat([images[k][v] for k in images for v in images[k]])
    Y = torch.cat([torch.full((len(images[k][v]),), v - 1) for k in images for v in images[k]])
    opt = torch.optim.Adam(reader.parameters(), lr=2e-3)
    for _ in range(steps):
        i = torch.randint(len(X), (128,), generator=g)
        loss = F.cross_entropy(reader(X[i]), Y[i]); opt.zero_grad(); loss.backward(); opt.step()
    return len(X)

def reader_accuracy(reader, images):
    with torch.no_grad():
        X = torch.cat([images[k][v] for k in images for v in images[k]])
        Y = torch.cat([torch.full((len(images[k][v]),), v - 1) for k in images for v in images[k]])
        return round((reader(X).argmax(-1) == Y).float().mean().item(), 4)

def train(mode, iters=200, tasks=16, G=8, lr=2e-3, seed=0):
    torch.manual_seed(seed); random.seed(seed); rng = random.Random(seed)
    train_imgs = bank(150, seed=1); test_imgs = bank(10, seed=999)
    policy = Policy(symbolic="sym" in mode); info = {"mode": mode, "seed": seed}
    if "pre" in mode:
        info["reader_images"] = pretrain_reader(policy.reader, train_imgs, seed=seed)
    if "sft" in mode:
        info["sft_pairs"] = sft(policy, train_imgs, 10, seed=seed, freeze_reader="pre" in mode)
    info["reader_acc_test"] = reader_accuracy(policy.reader, test_imgs)
    reader_lr = 0.0 if "frozen" in mode else (1e-4 if "pre" in mode else lr)
    opt = torch.optim.Adam([{"params": policy.mlp.parameters(), "lr": lr},
                            {"params": policy.reader.parameters(), "lr": reader_lr}]); log = []; dead = total = 0; t0 = time.time()
    for it in range(iters + 1):
        if it % 25 == 0:
            row = {"iter": it, **evaluate(policy, test_imgs, n=600), "dead": round(dead / max(total, 1), 3),
                   "sec": round(time.time() - t0)}
            print(row, flush=True); log.append(row); dead = total = 0
        if it == iters: break
        loss = 0.0
        for _ in range(tasks):
            task = tuple(rng.randint(1, HI) for _ in range(3))
            group = [rollout(policy, task, train_imgs, rng) for _ in range(G)]
            r = torch.tensor([x[2] for x in group]); adv = (r - r.mean()) / (r.std() + 1e-6)
            dead += bool(r.std() < 1e-6); total += 1
            loss = loss - (adv * torch.stack([x[1] for x in group])).mean()
        opt.zero_grad(); (loss / tasks).backward(); opt.step()
    info["reader_acc_test_after"] = reader_accuracy(policy.reader, test_imgs)
    print(info, flush=True)
    json.dump({"info": info, "log": log}, open(f"log_{mode}_s{seed}.json", "w"))
    torch.save(policy.state_dict(), f"policy_{mode}_s{seed}.pt")

if __name__ == "__main__":
    train(sys.argv[1], seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0,
          iters=int(sys.argv[3]) if len(sys.argv) > 3 else 200)
