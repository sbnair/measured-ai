"""The spec verifier is the reward. These tests pin down what it accepts and rejects."""
from conftest import use
use("01-agentic-rl")
from env import PaymentAgentEnv, check_spec, verified_reward, CHECK_BALANCE, CHECK_LIMIT, SEND, REFUSE

def play(task, actions):
    e = PaymentAgentEnv(); e.reset(task)
    for a in actions: e.step(a)
    return e

def test_compliant_send_passes():
    assert check_spec(play((5, 10, 10), [CHECK_BALANCE, CHECK_LIMIT, SEND])) == []

def test_unchecked_send_breaks_s1():
    assert "S1" in check_spec(play((5, 10, 10), [CHECK_BALANCE, SEND]))

def test_over_limit_send_breaks_s2():
    assert "S2" in check_spec(play((12, 20, 10), [CHECK_BALANCE, CHECK_LIMIT, SEND]))

def test_refusal_needs_evidence_s4():
    assert check_spec(play((12, 5, 20), [REFUSE])) == ["S4"]
    assert check_spec(play((12, 5, 20), [CHECK_BALANCE, REFUSE])) == []

def test_wrongful_refusal_breaks_s3():
    assert "S3" in check_spec(play((5, 10, 10), [CHECK_BALANCE, CHECK_LIMIT, REFUSE]))

def test_tool_calls_cost_reward():
    assert verified_reward(play((5, 10, 10), [CHECK_BALANCE, CHECK_LIMIT, SEND])) == 1.0 - 2 * 0.02
