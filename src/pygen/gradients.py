import torch
import functools

# TODO instead of treating named tuple as a special case; allow users to
# register their classes, and consider requiring users to register their
# classes created with namedtuple factory

def unroll_torch_tensors(value, detach=False, get_grad_instead=False):
    if detach and get_grad_instead:
        raise RuntimeError('Can only set one of detach and get_grad_instead')

    def recurse(x):
        return unroll_torch_tensors(x, detach=detach, get_grad_instead=get_grad_instead)

    if isinstance(value, torch.Tensor):
        if get_grad_instead:
            return (value.grad,)
        elif detach:
            return (value.detach(),)
        else:
            return (value,)
    if isinstance(value, tuple):
        # NOTE: this also handles namedtuples
        result = functools.reduce(lambda a, b: a + b, map(recurse, value), ())
        assert isinstance(result, tuple)
        return result
    elif isinstance(value, list):
        result = functools.reduce(lambda a, b: a + b, map(recurse, value), ())
        assert isinstance(result, tuple)
        return result
    elif isinstance(value, dict):
        result = functools.reduce(lambda a, b: a + b, map(recurse, value.values()), ())
        assert isinstance(result, tuple)
        return result
    else:
        raise NotImplementedError(f'gradient for object {value} not implemented')

def _isnamedtuple(value):
    return isinstance(value, tuple) and getattr(value, '_fields', None)

def _roll_torch_tensors(value, unrolled, start_idx):
    if isinstance(value, torch.Tensor):
        return unrolled[start_idx], 1
    if isinstance(value, tuple) or isinstance(value, list):
        rolled = []
        num = 0
        for element in value:
            rolled_element, element_num = _roll_torch_tensors(element, unrolled, start_idx)
            start_idx += element_num
            num += element_num
            rolled.append(rolled_element)
        if isinstance(value, list):
            return rolled, num
        elif _isnamedtuple(value):
            return type(value)(*rolled), num
        else:
            return tuple(rolled), num
    elif isinstance(value, dict):
        rolled = {}
        num = 0
        # NOTE: order is guaranteed to be the same as order for value.values() when unrolling
        for k, element in value.items():
            rolled_element, element_num = _roll_torch_tensors(element, unrolled, start_idx)
            start_idx += element_num
            num += element_num
            rolled[k] = rolled_element
        return rolled, num
    else:
        raise NotImplementedError(f'gradient for object {value} not implemented')


def roll_torch_tensors(value, unrolled):
    rolled, num = _roll_torch_tensors(value, unrolled, 0)
    assert num == len(unrolled)
    return rolled


def track(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().requires_grad_(True)
    if _isnamedtuple(value):
        return type(value)(*map(track, value))
    elif isinstance(value, tuple):
        return tuple(map(track, value))
    elif isinstance(value, list):
        return list(map(track, value))
    elif isinstance(value, dict):
        d = {}
        for k, element in value.items():
            d[k] = track(element)
        return d
    else:
        raise NotImplementedError(f'gradient for object {value} not implemented')
