"""DPO loss from sequence log-probabilities, with a sanity check and one gradient step."""
import math
import torch
import torch.nn.functional as F


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    """pi_* / ref_*: summed log-probs of the chosen (w) and rejected (l) responses."""
    reward_w = beta * (pi_w - ref_w)          # implicit reward of the chosen answer
    reward_l = beta * (pi_l - ref_l)          # implicit reward of the rejected answer
    loss = -F.logsigmoid(reward_w - reward_l).mean()
    return loss, reward_w.detach(), reward_l.detach()


torch.manual_seed(0)
ref_w = torch.tensor([-42.0, -55.0, -31.0, -60.0])   # reference log-probs, 4 preference pairs
ref_l = torch.tensor([-40.0, -58.0, -30.0, -57.0])

# 1) At initialisation the policy equals the reference, so the loss must be log 2.
loss, _, _ = dpo_loss(ref_w.clone(), ref_l.clone(), ref_w, ref_l)
print(f"loss at init = {loss.item():.4f}   (log 2 = {math.log(2):.4f})")

# 2) Treat the policy log-probs as parameters and take gradient steps.
pi_w = ref_w.clone().requires_grad_(True)
pi_l = ref_l.clone().requires_grad_(True)
opt = torch.optim.SGD([pi_w, pi_l], lr=5.0)
for step in range(1, 4):
    opt.zero_grad()
    loss, rw, rl = dpo_loss(pi_w, pi_l, ref_w, ref_l)
    loss.backward()
    print(f"step {step}: loss = {loss.item():.4f}   "
          f"grad chosen = {pi_w.grad[0].item():+.4f}   grad rejected = {pi_l.grad[0].item():+.4f}")
    opt.step()

loss, rw, rl = dpo_loss(pi_w, pi_l, ref_w, ref_l)
print(f"after 3 steps: loss = {loss.item():.4f}   mean reward margin = {(rw - rl).mean().item():.4f}")
