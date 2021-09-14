import torch
import functools

def unroll_torch_tensors(value):
    # used to unroll the retval grad for the 'grad_outputs' argument to torch.autograd.grad in dml/lang.py
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, tuple):
        result = functools.reduce(lambda a, b: a + b, map(unroll_torch_tensors, value))
        assert isinstance(result, tuple)
        return result
    elif isinstance(value, list):
        result = functools.reduce(lambda a, b: a + b, map(unroll_torch_tensors, value))
        assert isinstance(result, tuple)
        return result
    else:
        raise NotImplementedError(f'gradient for object {value} not implemented')

# # needed to unroll the retval grad for the 'grad_outputs' argument to torch.autograd.grad in dml/lang.py
# # TODO: may not be needed if we set only_inputs=False
# def roll_torch_tensors(value):
