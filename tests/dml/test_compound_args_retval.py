from pygen.dml.lang import gendml
from pygen import gentrace
from pygen.choice_address import addr

import torch


@gendml
def f1(x, y, compound):
    (a, b, (c, d, (e, g))) = compound
    return [(x, y), (x, [x, y]), a + b + c + d + e + g]


def test_choice_gradients_no_calls():
    t = torch.tensor
    x = y = a = b = c = d = e = g = t(0.0)
    compound = (a, b, [c, d, (e, g)])
    trace = f1.simulate((x, y, compound))
    compound_retgrad = (t(1.0), t(1.0), [t(1.0), t(1.0), (t(1.0), t(1.0))])
    retgrad = [(t(1.0), t(1.0)), (t(1.0), [t(1.0), t(1.0)]), compound_retgrad]
    expected_x_grad = t(3.0)
    expected_y_grad = t(2.0)
    expected_abcdeg_grad = t(1.0)
    (arg_grads, choice_values, choice_grads) = trace.choice_gradients(None, retgrad)
    assert len(arg_grads) == 3
    x_grad, y_grad, compound_grad = arg_grads
    (a_grad, b_grad, [c_grad, d_grad, (e_grad, g_grad)]) = compound_grad
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
    compound = (a, b, [c, d, (e, g)])
    trace = f2.simulate((x, y, compound))
    compound_retgrad = (t(1.0), t(1.0), [t(1.0), t(1.0), (t(1.0), t(1.0))])
    retgrad = [(t(1.0), t(1.0)), (t(1.0), [t(1.0), t(1.0)]), compound_retgrad]
    retgrad = (retgrad, retgrad)
    expected_x_grad = t(3.0) * 2  # 0
    expected_y_grad = t(2.0) * 2 # 5
    expected_abcdeg_grad = t(1.0) * 2
    (arg_grads, choice_values, choice_grads) = trace.choice_gradients(None, retgrad)
    print(arg_grads)
    assert len(arg_grads) == 3
    x_grad, y_grad, compound_grad = arg_grads
    (a_grad, b_grad, [c_grad, d_grad, (e_grad, g_grad)]) = compound_grad
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