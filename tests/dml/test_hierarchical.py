from pygen.choice_address import addr
from pygen.choice_trie import MutableChoiceTrie
from pygen.dml.lang import gendml
from pygen.dists import bernoulli
from pygen import gentrace
import torch

torch_false = torch.tensor(0.0)
torch_true = torch.tensor(1.0)


@gendml
def f(prob):
    if gentrace(bernoulli, (prob,), addr('done')):
        return torch.tensor(1)
    else:
        left_result = gentrace(f, (prob,), addr('left'))
        right_result = gentrace(f, (prob,), addr('right'))
        return left_result + right_result + torch.tensor(1)


def count_calls(trie):
    if trie[addr('done')]:
        return 1
    else:
        left_trie = trie.get_subtrie(addr('left'))
        right_trie = trie.get_subtrie(addr('right'))
        return 1 + count_calls(left_trie) + count_calls(right_trie)


def test_simulate():
    prob = torch.tensor(0.8)
    trace = f.simulate((prob,))
    trie = trace.get_choice_trie()
    num_dones = count_calls(trie)
    assert num_dones == trace.get_retval()


def get_initial_choice_trie():
    trie = MutableChoiceTrie()
    trie[addr('done')] = torch_false
    trie[addr('left', 'done')] = torch_false
    trie[addr('right', 'done')] = torch_true
    trie[addr('left', 'left', 'done')] = torch_true
    trie[addr('left', 'right', 'done')] = torch_true
    return (trie, 5)


def test_generate():
    prob = torch.tensor(0.8)
    (trie, num_choices) = get_initial_choice_trie()
    (trace, log_weight) = f.generate((prob,), trie)
    assert trace.get_args() == (prob,)
    assert trace.get_retval() == num_choices
    assert trace.get_choice_trie() == trie
    expected_score = torch.log((1-prob) * (1-prob) * prob * prob * prob)
    assert torch.isclose(expected_score, trace.get_score())
    assert torch.isclose(expected_score, log_weight)


def test_update():
    prob = torch.tensor(0.8)
    (init, _) = get_initial_choice_trie()
    (trace, _) = f.generate((prob,), init)

    constraints = MutableChoiceTrie()
    constraints[addr('left', 'done')] = torch_true
    (new_trace, log_weight, discard) = trace.update((prob,), constraints)

    expected_discard = MutableChoiceTrie()
    expected_discard[addr('left', 'done')] = torch_false
    expected_discard[addr('left', 'left', 'done')] = torch_true
    expected_discard[addr('left', 'right', 'done')] = torch_true
    assert discard == expected_discard

    expected_choice_trie = MutableChoiceTrie()
    expected_choice_trie[addr('done')] = torch_false
    expected_choice_trie[addr('left', 'done')] = torch_true
    expected_choice_trie[addr('right', 'done')] = torch_true
    assert new_trace.get_choice_trie() == expected_choice_trie

    assert torch.isclose(log_weight, new_trace.get_score() - trace.get_score())

    (round_trip_trace, negative_log_weight, round_trip_discard) = new_trace.update((prob,), discard)
    assert round_trip_trace.get_choice_trie() == trace.get_choice_trie()
    assert round_trip_discard == constraints
    assert torch.isclose(negative_log_weight, -log_weight)


test_generate()
