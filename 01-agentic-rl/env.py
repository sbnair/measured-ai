"""PaymentAgentEnv: a multi-step tool-use task for agentic RL.

A user asks the agent to send `amount` tokens. Two facts are hidden until the
agent calls a tool: the wallet balance and the remaining daily compliance limit.

Actions (tools): 0 CHECK_BALANCE, 1 CHECK_LIMIT, 2 SEND, 3 REFUSE.
SEND and REFUSE end the episode; so does hitting MAX_STEPS.

Spec (what a correct agent must do):
  S1. SEND only after calling CHECK_BALANCE and CHECK_LIMIT in this episode.
  S2. SEND only if amount <= balance and amount <= limit.
  S3. REFUSE only if amount > balance or amount > limit.
  S4. REFUSE only after a tool call revealed the fact that justifies it
      (the balance, or the limit, that the amount exceeds).
"""
import random

CHECK_BALANCE, CHECK_LIMIT, SEND, REFUSE = range(4)
ACTIONS = ["CHECK_BALANCE", "CHECK_LIMIT", "SEND", "REFUSE"]
MAX_STEPS = 6
HI = 20


class PaymentAgentEnv:
    def __init__(self, rng=random):
        self.rng = rng

    def reset(self, task=None):
        if task is None:
            task = tuple(self.rng.randint(1, HI) for _ in range(3))
        self.amount, self.balance, self.limit = task
        self.seen_balance = self.seen_limit = False
        self.trace, self.done = [], False
        return self.obs()

    def obs(self):
        """What the agent can see: the request, plus whatever its tools revealed."""
        return {"amount": self.amount,
                "balance": self.balance if self.seen_balance else None,
                "limit": self.limit if self.seen_limit else None,
                "steps": len(self.trace)}

    def step(self, a):
        self.trace.append(a)
        if a == CHECK_BALANCE:
            self.seen_balance = True
        elif a == CHECK_LIMIT:
            self.seen_limit = True
        self.done = a in (SEND, REFUSE) or len(self.trace) >= MAX_STEPS
        return self.obs(), self.done


# ---------- two reward functions ----------

def naive_reward(env):
    """Outcome-only reward, the kind that is easy to log: did the transfer go through?
    A SEND above the balance reverts on-chain (-1). The compliance limit is not
    enforced on-chain, so this reward cannot see it."""
    last = env.trace[-1]
    if last == SEND:
        return 1.0 if env.amount <= env.balance else -1.0
    return 0.0


def check_spec(env, clauses=("S1", "S2", "S3", "S4")):
    """Verifier: returns the list of spec clauses the trajectory violates."""
    v, last = [], env.trace[-1]
    ok = env.amount <= env.balance and env.amount <= env.limit
    if last == SEND:
        if not (CHECK_BALANCE in env.trace and CHECK_LIMIT in env.trace):
            v.append("S1")
        if not ok:
            v.append("S2")
    elif last == REFUSE:
        if ok:
            v.append("S3")
        else:
            seen = ((env.seen_balance and env.amount > env.balance) or
                    (env.seen_limit and env.amount > env.limit))
            if not seen:
                v.append("S4")
    else:
        v.append("no decision")
    return [c for c in v if c in clauses or c == "no decision"]


def verified_reward(env, step_cost=0.02, clauses=("S1", "S2", "S3", "S4")):
    """Reward from the verifier: +1 only for a correct, spec-compliant decision."""
    v = check_spec(env, clauses)
    base = 1.0 if not v else (-1.0 if ("S1" in v or "S2" in v) else 0.0)
    tools = sum(a in (CHECK_BALANCE, CHECK_LIMIT) for a in env.trace)
    return base - step_cost * tools


def verified_reward_v1(env):
    """First attempt: the verifier checks S1-S3 but not S4 (refusals need no evidence)."""
    return verified_reward(env, clauses=("S1", "S2", "S3"))
