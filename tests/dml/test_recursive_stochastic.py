from pygen.dml.lang import gendml, inline
from pygen.dists import bernoulli
from pygen.choice_address import addr
from pygen.choice_trie import ChoiceTrie, MutableChoiceTrie
import torch

DONE = 'done'


@gendml
def f(prob, i):
    done = bernoulli(prob) @ addr((DONE, i))
    if done:
        return i
    else:
        return f(prob, i+1) @ inline


@gendml
def g(prob):
    return f(prob, 1) @ inline


def check_choices(n, choices):
    assert n == len(choices.asdict())
    for i in range(1, n):
        a = addr((DONE, i))
        assert not choices[a]
    a = addr((DONE, n))
    assert choices[a]


def get_expected_score(prob, n):
    return torch.log(1.0 - torch.tensor(prob)) * (n-1) + torch.log(torch.tensor(prob))


def test_simulate():
    prob = 0.4
    trace = g.simulate((prob,))
    assert trace.get_gen_fn() == g
    assert trace.get_args() == (prob,)
    n = trace.get_retval()
    choices = trace.get_choice_trie()
    assert torch.isclose(trace.get_score(), get_expected_score(prob, n))
    check_choices(n, choices)


def test_generate():
    prob = 0.4

    # fully constrained
    trie = MutableChoiceTrie()
    trie[addr((DONE, 1))] = torch.tensor(0.0)
    trie[addr((DONE, 2))] = torch.tensor(1.0)
    (trace, log_weight) = g.generate((prob,), trie)
    assert trace.get_gen_fn() == g
    assert trace.get_args() == (prob,)
    n = trace.get_retval()
    choices = trace.get_choice_trie()
    check_choices(n, choices)
    assert torch.isclose(trace.get_score(), get_expected_score(prob, n))
    assert isinstance(log_weight, torch.Tensor)
    assert not log_weight.requires_grad
    assert torch.isclose(log_weight, trace.get_score())

    # not fully constrained
    trie = MutableChoiceTrie()
    trie[addr((DONE, 1))] = torch.tensor(0.0)
    (trace, log_weight) = g.generate((prob,), trie)
    assert trace.get_gen_fn() == g
    assert trace.get_args() == (prob,)
    n = trace.get_retval()
    choices = trace.get_choice_trie()
    assert n == len(choices.asdict())
    assert torch.isclose(trace.get_score(), get_expected_score(prob, n))
    check_choices(n, choices)
    assert isinstance(log_weight, torch.Tensor)
    assert not log_weight.requires_grad
    expected_log_weight = torch.log(1.0 - torch.tensor(prob))
    assert torch.isclose(log_weight, expected_log_weight)


def test_update():
    prob = 0.4

    trie = MutableChoiceTrie()
    trie[addr((DONE, 1))] = torch.tensor(1.0)
    (trace, _) = g.generate((prob,), trie)
    new_prob = 0.45

    trie = MutableChoiceTrie()
    trie[addr((DONE, 1))] = torch.tensor(0.0)
    trie[addr((DONE, 2))] = torch.tensor(1.0)
    (new_trace, log_weight, discard) = trace.update((new_prob,), trie)

    assert new_trace.get_gen_fn() == g
    assert new_trace.get_args() == (new_prob,)
    n = new_trace.get_retval()
    assert n == 2
    choices = new_trace.get_choice_trie()
    check_choices(n, choices)
    assert torch.isclose(new_trace.get_score(), get_expected_score(new_prob, n))

    assert isinstance(log_weight, torch.Tensor)
    assert not log_weight.requires_grad
    assert torch.isclose(log_weight, new_trace.get_score() - trace.get_score())

    assert isinstance(discard, ChoiceTrie)
    assert len(discard.asdict()) == 1
    assert discard[addr((DONE, 1))] == torch.tensor(1.0)

test_generate()
test_update()
