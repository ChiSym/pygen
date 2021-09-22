from pygen.dml.lang import gendml
from pygen import gentrace
from pygen.choice_address import addr
from pygen.gradients import supports_ad

import torch

from collections import namedtuple
Foo = namedtuple('Foo', ['x', 'y', 'z'])

def unroll_bar(value, detach, get_grad_instead):
    assert isinstance(value, Bar)
    assert not (detach and get_grad_instead)
    if detach:
        return (value.get_x().detach(), value.get_y().detach(), value.get_z().detach())
    elif get_grad_instead:
        return (value.get_x().grad, value.get_y().grad, value.get_z().grad)
    else:
        return (value.get_x(), value.get_y(), value.get_z())
    
def roll_bar(value, unrolled, start_idx):
    assert isinstance(value, Bar)
    x, y, z = unrolled[start_idx], unrolled[start_idx+1], unrolled[start_idx+2]
    return Bar(x, y, z), 3

@supports_ad(unroll_bar, roll_bar)
class Bar:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z
    def get_x(self):
        return self._x
    def get_y(self):
        return self._y
    def get_z(self):
        return self._z


@gendml
def f1(x, y, compound):
    assert isinstance(compound, dict)
    (a, b, (c, d, (e, g))) = compound['k1']
    return [(x, y), (x, [x, y]), {'k2': a + b + c + d + e + g}, Foo(x, x, y), Bar(x, y, y)]


def test_choice_gradients_no_calls():
    t = torch.tensor
    x = y = a = b = c = d = e = g = t(0.0)
    compound = {'k1': (a, b, [c, d, (e, g)])}
    trace = f1.simulate((x, y, compound))
    retgrad = [(t(1.0), t(1.0)), (t(1.0), [t(1.0), t(1.0)]), {'k2': t(1.0)}, Foo(t(1.1), t(1.2), t(1.3)), Bar(t(1.4), t(1.5), t(1.6))]
    expected_x_grad = t(3.0) + t(1.1) + t(1.2) + t(1.4)
    expected_y_grad = t(2.0) + t(1.3) + t(1.5) + t(1.6)
    expected_abcdeg_grad = t(1.0)
    (arg_grads, choice_values, choice_grads) = trace.choice_gradients(None, retgrad)
    assert len(arg_grads) == 3
    x_grad, y_grad, compound_grad = arg_grads
    (a_grad, b_grad, [c_grad, d_grad, (e_grad, g_grad)]) = compound_grad['k1']
    assert torch.allclose(x_grad, expected_x_grad)
    assert torch.allclose(y_grad, expected_y_grad)
    assert torch.allclose(a_grad, expected_abcdeg_grad)
    assert torch.allclose(b_grad, expected_abcdeg_grad)
    assert torch.allclose(c_grad, expected_abcdeg_grad)
    assert torch.allclose(d_grad, expected_abcdeg_grad)
    assert torch.allclose(e_grad, expected_abcdeg_grad)
    assert torch.allclose(g_grad, expected_abcdeg_grad)
    assert choice_values is None
    assert choice_grads is None


@gendml
def f2(x, y, compound):
    z1 = gentrace(f1, (x, y, compound))  # spliced call
    z2 = gentrace(f1, (x, y, compound), addr('z')) # not spliced call
    return z1, z2


def test_choice_gradients_calls():
    t = torch.tensor
    x = y = a = b = c = d = e = g = t(0.0)
    compound = {'k1': (a, b, [c, d, (e, g)])}
    trace = f2.simulate((x, y, compound))
    retgrad = [(t(1.0), t(1.0)), (t(1.0), [t(1.0), t(1.0)]), {'k2': t(1.0)}, Foo(t(1.1), t(1.2), t(1.3)), Bar(t(1.4), t(1.5), t(1.6))]
    retgrad = (retgrad, retgrad)
    expected_x_grad = (t(3.0) + t(1.1) + t(1.2) + t(1.4))* 2
    expected_y_grad = (t(2.0) + t(1.3) + t(1.5) + t(1.6)) * 2
    expected_abcdeg_grad = t(1.0) * 2
    (arg_grads, choice_values, choice_grads) = trace.choice_gradients(None, retgrad)
    print(arg_grads)
    assert len(arg_grads) == 3
    x_grad, y_grad, compound_grad = arg_grads
    (a_grad, b_grad, [c_grad, d_grad, (e_grad, g_grad)]) = compound_grad['k1']
    assert torch.allclose(x_grad, expected_x_grad)
    assert torch.allclose(y_grad, expected_y_grad)
    assert torch.allclose(a_grad, expected_abcdeg_grad)
    assert torch.allclose(b_grad, expected_abcdeg_grad)
    assert torch.allclose(c_grad, expected_abcdeg_grad)
    assert torch.allclose(d_grad, expected_abcdeg_grad)
    assert torch.allclose(e_grad, expected_abcdeg_grad)
    assert torch.allclose(g_grad, expected_abcdeg_grad)
    assert choice_values is None
    assert choice_grads is None
