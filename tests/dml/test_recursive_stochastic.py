from pygen.dml.lang import gendml
from pygen.dists import bernoulli, normal
import torch


@gendml
def f(prob, i):
    done = gentrace(bernoulli, (prob,), ("done", i))
    if done:
        return i
    else:
        return gentrace(f, (prob, i+1))

@gendml
def g(prob):
    return gentrace(f, (prob, 1))

def check_choices(n, choices):
    assert n == len(choices)
    for i in range(1, n):
        addr = ("done", i)
        assert addr in choices
        assert not choices[addr]
    addr = ("done", n)
    assert addr in choices
    assert choices[addr]

def get_expected_score(prob, n):
    return torch.log(1.0 - torch.tensor(prob)) * (n-1) + torch.log(torch.tensor(prob))

def test_simulate():
    prob = 0.4
    trace = g.simulate((prob,))

    assert trace.get_gen_fn() == g
    assert trace.get_args() == (prob,)
    n = trace.get_retval()
    choices = trace.get_choices()
    assert n == len(choices)
    assert torch.isclose(trace.get_score(), get_expected_score(prob, n))
    for i in range(1, n):
        addr = ("done", i)
        assert addr in choices
        assert not choices[addr]
    addr = ("done", n)
    assert addr in choices
    assert choices[addr]

def test_generate():
    prob = 0.4

    # fully constrained
    (trace, log_weight) = g.generate((prob,), {("done", 1) : torch.tensor(0.0), ("done", 2) : torch.tensor(1.0)})
    assert trace.get_gen_fn() == g
    assert trace.get_args() == (prob,)
    n = trace.get_retval()
    choices = trace.get_choices()
    check_choices(n, choices)
    assert torch.isclose(trace.get_score(), get_expected_score(prob, n))
    assert isinstance(log_weight, torch.Tensor)
    assert not log_weight.requires_grad
    assert torch.isclose(log_weight, trace.get_score())

    # not fully constrained
    (trace, log_weight) = g.generate((prob,), {("done", 1) : torch.tensor(0.0)})
    assert trace.get_gen_fn() == g
    assert trace.get_args() == (prob,)
    n = trace.get_retval()
    choices = trace.get_choices()
    assert n == len(choices)
    assert torch.isclose(trace.get_score(), get_expected_score(prob, n))
    check_choices(n, choices)
    assert isinstance(log_weight, torch.Tensor)
    assert not log_weight.requires_grad
    expected_log_weight = torch.log(1.0 - torch.tensor(prob))
    assert torch.isclose(log_weight, expected_log_weight)

def test_update():
    prob = 0.4
    (trace, _) = g.generate((prob,), {("done", 1) : torch.tensor(1.0)})
    new_prob = 0.45
    (new_trace, log_weight, discard) = trace.update((new_prob,), {
        ("done", 1) : torch.tensor(0.0),
        ("done", 2) : torch.tensor(1.0)})

    assert new_trace.get_gen_fn() == g
    assert new_trace.get_args() == (new_prob,)
    n = new_trace.get_retval()
    assert n == 2
    choices = new_trace.get_choices()
    check_choices(n, choices)
    assert torch.isclose(new_trace.get_score(), get_expected_score(new_prob, n))

    assert isinstance(log_weight, torch.Tensor)
    assert not log_weight.requires_grad
    assert torch.isclose(log_weight, new_trace.get_score() - trace.get_score())

    print(discard)
    assert isinstance(discard, dict)
    assert len(discard) == 1
    assert ("done", 1) in discard
    assert discard[("done", 1)] == torch.tensor(1.0)
