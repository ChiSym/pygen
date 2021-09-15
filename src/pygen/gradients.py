import torch
import functools


# TODO support dict

def unroll_torch_tensors(value, detach=False):
    def recurse(x):
        return unroll_torch_tensors(x, detach=detach)
    if isinstance(value, torch.Tensor):
        if detach:
            return (value.detach(),)
        else:
            return (value,)
    if isinstance(value, tuple):
        result = functools.reduce(lambda a, b: a + b, map(recurse, value))
        assert isinstance(result, tuple)
        return result
    elif isinstance(value, list):
        result = functools.reduce(lambda a, b: a + b, map(recurse, value))
        assert isinstance(result, tuple)
        return result
    else:
        raise NotImplementedError(f'gradient for object {value} not implemented')


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
        if isinstance(value, tuple):
            return tuple(rolled), num
        else:
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
    if isinstance(value, tuple):
        return tuple(map(track, value))
    elif isinstance(value, list):
        return list(map(track, value))
    else:
        raise NotImplementedError(f'gradient for object {value} not implemented')