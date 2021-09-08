from pygen.dml.lang import gendml
from pygen.dists import bernoulli, normal
from pygen import gentrace
import torch

@gendml
def f(a, b):
    return a + b

def test_call():
    assert f(1, 3) == 4
    assert f('a', 'b') == 'ab'
    assert f([1], [2]) == [1, 2]

def test_simulate():

    t = torch.tensor(2)
    trace = f.simulate((1, t))

    assert trace.get_gen_fn() == f
    assert trace.get_args() == (1, t)
    assert not trace.get_args()[1].requires_grad
    assert trace.get_retval() == 3
    assert isinstance(trace.get_retval(), torch.Tensor)
    assert not trace.get_retval().requires_grad
    assert isinstance(trace.get_score(), torch.Tensor)
    assert not trace.get_score().requires_grad
    assert trace.get_score() == 0.0
    assert not trace.get_retval().requires_grad
    assert isinstance(trace.get_choices(), dict)
    assert len(trace.get_choices()) == 0

def test_generate():

    # check that arguments can be both Python numbers and torch.Tensor
    t = torch.tensor(2)
    (trace, log_weight) = f.generate((1, t), {})

    assert trace.get_gen_fn() == f
    assert trace.get_args() == (1, t)
    assert not trace.get_args()[1].requires_grad
    assert trace.get_retval() == 3
    assert isinstance(trace.get_retval(), torch.Tensor)
    assert not trace.get_retval().requires_grad
    assert isinstance(trace.get_score(), torch.Tensor)
    assert not trace.get_score().requires_grad
    assert trace.get_score() == 0.0
    assert not trace.get_retval().requires_grad
    assert isinstance(trace.get_choices(), dict)
    assert len(trace.get_choices()) == 0

def test_update():

    t = torch.tensor(2)
    trace = f.simulate((1, t))
    t_new = torch.tensor(3)
    (new_trace, log_weight, discard) = trace.update((2, t_new), {})

    assert new_trace.get_gen_fn() == f
    assert new_trace.get_args() == (2, t_new)
    assert not new_trace.get_args()[1].requires_grad
    assert new_trace.get_retval() == 5
    assert isinstance(new_trace.get_retval(), torch.Tensor)
    assert not new_trace.get_retval().requires_grad
    assert isinstance(new_trace.get_score(), torch.Tensor)
    assert not new_trace.get_score().requires_grad
    assert new_trace.get_score() == 0.0
    assert not new_trace.get_retval().requires_grad
    assert isinstance(new_trace.get_choices(), dict)
    assert len(new_trace.get_choices()) == 0

    assert isinstance(log_weight, torch.Tensor)
    assert not log_weight.requires_grad
    assert torch.isclose(log_weight, torch.tensor(0.0))

    assert isinstance(discard, dict)
    assert len(discard) == 0
