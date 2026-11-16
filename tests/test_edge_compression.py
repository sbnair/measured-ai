"""Fast correctness checks for the compression code: no trained weights needed."""
import torch
from conftest import use
use("03-edge-compression")
from model import GPT
import quant as Q, specdec as S

def test_kv_cache_matches_full_forward():
    torch.manual_seed(0); m = GPT(64, 4, 2).eval(); x = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        full = m(x)[:, -1]; c = S.new_caches(m); m(x[:, :15], c); part = m(x[:, 15:], c, start=15)[:, -1]
    assert (full - part).abs().max() < 1e-5

def test_greedy_speculative_equals_autoregressive():
    torch.manual_seed(1); t, d = GPT(64, 4, 2).eval(), GPT(32, 2, 1).eval(); p = torch.randint(0, 256, (1, 8))
    a, _ = S.autoregressive(t, p, 40, temp=0)
    for gamma in (1, 3, 5):
        b, _, _ = S.speculative(t, d, p, 40, gamma=gamma, temp=0)
        assert torch.equal(a, b)

def test_speculative_with_itself_accepts_everything():
    torch.manual_seed(2); t = GPT(64, 4, 2).eval(); p = torch.randint(0, 256, (1, 8))
    _, calls, alpha = S.speculative(t, t, p, 40, gamma=4, temp=0)
    assert alpha == 1.0 and calls == 8          # 40 tokens / (4 drafted + 1 bonus) per pass

def test_group_quant_error_bounded_by_half_step():
    torch.manual_seed(3); W = torch.randn(64, 128)
    for bits in (8, 4, 3):
        Wq = Q.fake_quant(W, bits, 32); step = W.reshape(64, 4, 32).abs().amax(-1, keepdim=True) / (2 ** (bits - 1) - 1)
        assert ((Wq - W).reshape(64, 4, 32).abs() <= step / 2 + 1e-6).all()

def test_2_4_pruning_pattern():
    torch.manual_seed(4); m = Q.prune_2_4(GPT(64, 4, 1))
    for _, lin in Q.linears(m):
        g = lin.weight.reshape(lin.weight.shape[0], -1, 4)
        assert ((g != 0).sum(-1) <= 2).all()

def test_gptq_not_worse_than_rtn_on_layer_output():
    torch.manual_seed(5); m = GPT(64, 4, 1).eval(); x = torch.randint(0, 256, (8, 32))
    with torch.no_grad():
        ref = m(x); e_rtn = (Q.rtn(m, 3, 32)(x) - ref).pow(2).mean(); e_gptq = (Q.gptq(m, x, 3, 32)(x) - ref).pow(2).mean()
    assert e_gptq <= e_rtn * 1.05
