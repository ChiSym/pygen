from pygen.address import addr
from pygen.choice_trie import MutableChoiceTrie
from pygen.dml.lang import gendml
from pygen.dists import bernoulli, normal
import torch

@gendml
def f(prob):
    if gentrace(bernoulli, (prob,), addr('done')):
        return 1
    else:
        left_result = gentrace(f, (prob,), addr('left'))
        right_result = gentrace(f, (prob,), addr('right'))
        return left_result + right_result + 1

torch_false = torch.tensor(0.0)
torch_true = torch.tensor(1.0)

def count_calls(trie):
    if trie[addr('done')]:
        return 1
    else:
        left_trie = trie.get_subtrie(addr('left'))
        right_trie = trie.get_subtrie(addr('right'))
        return 1 + count_calls(left_trie) + count_calls(right_trie)

def test_simulate():
    prob = 0.3
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
    prob = 0.3
    (trie, num_choices) = get_initial_choice_trie()
    (trace, log_weight) = f.generate((prob,), trie)
    assert trace.get_args() == (prob,)
    assert trace.get_retval() == num_choices
    assert trace.get_choice_trie() == trie
    expected_score = (1-prob) * (1-prob) * prob * prob * prob
    assert torch.isclose(expected_score, trace.get_score())
    assert torch.isclose(expected_score, log_weight)

def test_update():
    prob = 0.3
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

    expected_choice_trie = MutablechoiceTrie()
    expected_choice_trie[addr('done')] = torch_false
    expected_choice_trie[addr('left', 'done')] = torch_true
    expected_choice_trie[addr('right', 'done')] = torch_true
    assert new_trace.get_choices() == expected_choice_trie

    assert torch.isclose(log_weight, new_trace.get_score() - trace.get_score())
