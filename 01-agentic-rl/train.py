"""GRPO-style training of a small policy on PaymentAgentEnv. CPU, ~1 minute."""
import json, random, sys
import torch, torch.nn as nn
from env import PaymentAgentEnv, naive_reward, verified_reward, verified_reward_v1, check_spec, HI, MAX_STEPS, SEND, REFUSE

def featurize(o):
    b, l = o["balance"], o["limit"]
    return torch.tensor([o["amount"] / HI,
                         (b or 0) / HI, float(b is not None),
                         (l or 0) / HI, float(l is not None),
                         o["steps"] / MAX_STEPS,
                         float(b is not None and o["amount"] <= b),
                         float(l is not None and o["amount"] <= l)])

class Policy(nn.Module):
    def __init__(self, d=8, h=64, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, n_actions))
    def forward(self, x):
        return torch.distributions.Categorical(logits=self.net(x))

def rollout(policy, task, reward_fn):
    env = PaymentAgentEnv(); o = env.reset(task); logps, ents = [], []; done = False
    while not done:
        dist = policy(featurize(o)); a = dist.sample()
        logps.append(dist.log_prob(a)); ents.append(dist.entropy()); o, done = env.step(a.item())
    return env, torch.stack(logps).sum(), reward_fn(env), torch.stack(ents).mean()

def evaluate(policy, n=2000, seed=123):
    rng = random.Random(seed); stats = dict(correct=0, over_limit_send=0, overdraft_send=0, unchecked_send=0, unjustified_refuse=0, no_decision=0, tools=0)
    with torch.no_grad():
        for _ in range(n):
            task = tuple(rng.randint(1, HI) for _ in range(3))
            env = rollout(policy, task, naive_reward)[0]
            v = check_spec(env)
            stats["correct"] += not v
            last = env.trace[-1]
            if last == SEND:
                stats["overdraft_send"] += env.amount > env.balance
                stats["over_limit_send"] += env.amount <= env.balance and env.amount > env.limit
                stats["unchecked_send"] += "S1" in v
            stats["unjustified_refuse"] += last == REFUSE and ("S3" in v or "S4" in v)
            stats["no_decision"] += last not in (SEND, REFUSE)
            stats["tools"] += sum(a < 2 for a in env.trace)
    return {k: round(val / n, 4) for k, val in stats.items()}

def expert_action(o):
    """Scripted demonstrator: look up what you need, then decide."""
    if o["balance"] is None: return 0                      # CHECK_BALANCE
    if o["amount"] > o["balance"]: return 3                # REFUSE, with evidence
    if o["limit"] is None: return 1                        # CHECK_LIMIT
    return 3 if o["amount"] > o["limit"] else 2            # REFUSE or SEND

def sft_warm_start(policy, n_demos, epochs=30, lr=3e-3, seed=0):
    """Behaviour cloning on n_demos expert episodes: the cold start before RL."""
    rng = random.Random(seed); X, Y = [], []
    for _ in range(n_demos):
        env = PaymentAgentEnv(); o = env.reset(tuple(rng.randint(1, HI) for _ in range(3))); done = False
        while not done:
            a = expert_action(o); X.append(featurize(o)); Y.append(a); o, done = env.step(a)
    X, Y = torch.stack(X), torch.tensor(Y); opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(epochs):
        loss = -policy(X).log_prob(Y).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return len(Y)

def train(reward_fn, iters=300, tasks_per_iter=32, G=8, lr=3e-3, ent_coef=0.0, sft=0, seed=0):
    torch.manual_seed(seed); random.seed(seed)
    policy = Policy()
    if sft:
        print("sft steps:", sft_warm_start(policy, sft))
    opt = torch.optim.Adam(policy.parameters(), lr=lr); log = []
    dead = total = 0; rsum = 0.0
    for it in range(iters + 1):
        if it % 25 == 0:
            row = {"iter": it, **evaluate(policy)}
            if total:
                row["dead_groups"] = round(dead / total, 4); row["train_reward"] = round(rsum / (total * G), 4)
            log.append(row); dead = total = 0; rsum = 0.0
        if it == iters: break
        loss = 0.0
        for _ in range(tasks_per_iter):
            task = tuple(random.randint(1, HI) for _ in range(3))
            group = [rollout(policy, task, reward_fn) for _ in range(G)]   # G rollouts, same task
            r = torch.tensor([g[2] for g in group])
            adv = (r - r.mean()) / (r.std() + 1e-6)                        # group-relative advantage
            dead += bool(r.std() < 1e-6); total += 1; rsum += r.sum().item()  # a group with one reward teaches nothing
            loss = loss - (adv * torch.stack([g[1] for g in group])).mean()
            loss = loss - ent_coef * torch.stack([g[3] for g in group]).mean()   # entropy bonus keeps groups diverse
        opt.zero_grad(); (loss / tasks_per_iter).backward(); opt.step()
    return policy, log

if __name__ == "__main__":
    which = sys.argv[1]
    fn = {"naive": naive_reward, "verified_v1": verified_reward_v1, "verified_v2": verified_reward}[which]
    kw = dict(a.split("=") for a in sys.argv[2:])
    kw = {k: (int(v) if k in ("G", "iters", "seed", "sft") else float(v)) for k, v in kw.items()}
    tag = "_".join([which] + [f"{k}{v}" for k, v in kw.items()])
    policy, log = train(fn, **kw)
    for row in log: print(row)
    json.dump(log, open(f"log_{tag}.json", "w"))
    torch.save(policy.state_dict(), f"policy_{tag}.pt")
