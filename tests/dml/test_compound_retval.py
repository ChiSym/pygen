from pygen.dml.lang import gendml
from pygen.dists import normal
from pygen.choice_address import addr
from pygen.choice_trie import MutableChoiceTrie
from pygen import gentrace

import torch


@gendml
def f(x, y):
    return [(x, y), (x, [x, y])]


def test_choice_gradients():
    x = torch.tensor(1.1)
    y = torch.tensor(1.2)
    z = torch.tensor(1.3)
    trie = MutableChoiceTrie()
    trie[addr('z')] = z
    (trace, _) = f.generate((x, y), trie)
    t = torch.tensor
    retgrad = [(t(1.0), t(1.0)), (t(1.0), [t(1.0), t(1.0)])]
    expected_x_grad = t(3.0)
    expected_y_grad = t(2.0)
    (arg_grads, choice_values, choice_grads) = trace.choice_gradients(None, retgrad)

    assert len(arg_grads) == 2
    x_grad, y_grad = arg_grads
    assert torch.allclose(x_grad, expected_x_grad)
    assert torch.allclose(y_grad, expected_y_grad)
    assert choice_values is None
    assert choice_grads is None