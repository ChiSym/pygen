from .gfi import Trace, GenFn, Same, Unknown
from .choice_trie import ChoiceTrie, MutableChoiceTrie
import torch

def _check_is_primitive_and_get_choice(choice_trie):
    if not choice_trie.has_choice():
        raise RuntimeError(f'choice_trie is not primitive: {choice_trie}')
    return choice_trie.get_choice()

class TorchDist(GenFn):
    pass

class TorchDistTrace(Trace):

    def __init__(self, gen_fn, args, dist, value, lpdf):
        assert isinstance(gen_fn, TorchDist)
        assert isinstance(value, torch.Tensor)
        assert not value.requires_grad
        self.gen_fn = gen_fn
        self.args = args
        self.value = value
        self.lpdf = lpdf
        self.dist = dist

    def get_gen_fn(self):
        return self.gen_fn

    def get_args(self):
        return self.args

    def get_score(self):
        return self.lpdf

    def get_retval(self):
        return self.value

    def get_choice_trie(self):
        trie = MutableChoiceTrie()
        trie.set_choice(self.value)
        return trie

    def update(self, args, args_change, choice_trie):
        args_unchanged = args_change == Same()  # Q: can we promote tuple of Same to Same?
        value_unchanged = not bool(choice_trie)
        if args_unchanged:
            new_dist = self.dist
        else:
            new_dist = self.gen_fn.get_dist_class()(*args)
        discard = MutableChoiceTrie()
        prev_value = self.value
        if value_unchanged:
            # choice_trie is empty
            value = prev_value
            ret_change = Same()
        else:
            # choice_trie is not empty
            assert isinstance(choice_trie, ChoiceTrie)
            value = _check_is_primitive_and_get_choice(choice_trie)
            discard.set_choice(prev_value)
            ret_change = Unknown()
        if args_unchanged and value_unchanged:
            new_lpdf = self.lpdf
            log_weight = 0.0
        else:
            new_lpdf = new_dist.log_prob(value).sum()
            prev_lpdf = self.lpdf
            log_weight = new_lpdf - prev_lpdf
        new_trace = TorchDistTrace(self.gen_fn, args, new_dist, value, new_lpdf)
        return new_trace, log_weight, ret_change, discard

    def regenerate(self, args=None, args_change=None, selection=None):
        raise NotImplementedError()

    # TODO: we will also probably want to have 'inlined' distributions
    # that don't go through this process, but instead directly add to PyTorch's
    # dynamic computation graph

    def choice_gradients(self, selection, retval_grad):
        if selection is not None:
            raise NotImplementedError()
        with torch.inference_mode(mode=False):
            value = self.value.detach()
            args_tracked = tuple(
                arg.detach().clone().requires_grad_(True) if isinstance(arg, torch.Tensor) else arg
                for arg in self.get_args())
            lpdf = self.dist.log_prob(value).sum()
            lpdf.backward(retain_graph=False)
            arg_grads = tuple(
                arg.grad if isinstance(arg, torch.Tensor) else None
                for arg in args_tracked)
        return arg_grads, None, None

    def accum_param_grads(self, retgrad, scale_factor):
        raise NotImplementedError()


def torch_dist_to_gen_fn(dist_class):

    class gen_fn_class(TorchDist):

        def __init__(self):
            self.dist_class = dist_class

        def __repr__(self):
            return f'pygen.dists.torch_dist_to_gen_fn({repr(dist_class)})'

        def get_dist_class(self):
            return self.dist_class

        def simulate(self, args):
            dist = dist_class(*args)
            value = dist.sample()
            lpdf = dist.log_prob(value).sum()
            return TorchDistTrace(self, args, dist, value, lpdf)
    
        def generate(self, args, choice_trie):
            dist = dist_class(*args)
            if choice_trie:
                value = _check_is_primitive_and_get_choice(choice_trie)
            else:
                value = dist.sample()
            lpdf = dist.log_prob(value).sum()
            if choice_trie:
                log_weight = lpdf
            else:
                log_weight = torch.tensor(0.0, requires_grad=False)
            return (TorchDistTrace(self, args, dist, value, lpdf), log_weight)

    return gen_fn_class()


normal = torch_dist_to_gen_fn(torch.distributions.normal.Normal)
bernoulli = torch_dist_to_gen_fn(torch.distributions.bernoulli.Bernoulli)
