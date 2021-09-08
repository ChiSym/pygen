from .gfi import Trace, GenFn
from .choice_trie import MutableChoiceTrie
import torch

class PyTorchDist(GenFn):
    pass

class PyTorchDistTrace(Trace):

    def __init__(self, gen_fn, args, value, lpdf):
        assert isinstance(gen_fn, PyTorchDist)
        self.gen_fn
        self.args = args
        self.value = value
        self.lpdf = lpdf

    def get_args(self):
        return self.args

    def get_score(self):
        return self.lpdf

    def get_retval(self):
        return self.value

    def get_choice_trie(self):
        return MutableChoiceTrie({(): self.value)

    def update(self, args, choice_trie):
        discard = MutableChoiceTrie()
        prev_value = self.value
        if not choice_trie:
            value = prev_value
        else:
            discard[()] = prev_value
            value = _check_is_primitive_and_get_value(choice_trie)
        new_dist = self.gen_fn.get_dist_class()(*args)
        new_lpdf = new_dist.log_prob(value).sum()
        prev_lpdf = self.lpdf
        log_weight = new_lpdf - prev_lpdf
        new_trace = PyTorchDistTrace(self.gen_fn, args, value, new_lpdf)
        return (new_trace, log_weight, discard)

    def regenerate(self, args, selection):
        raise NotImplementedError()

    # NOTE: as an optimization, these methods currently use the 'logpdf' method
    # (see below) instead..

    def accum_param_grads(self, retgrad, scale_factor):
        raise NotImplementedError()

    def choice_gradients(self, selection, retgrad):
        raise NotImplementedError()


def torch_dist_to_gen_fn(dist_class):

    class gen_fn_class(PyTorchDist):

        def __init__(self):
            self.dist_class = dist_class

        def get_dist_class(self):
            return self.dist_class

        def _check_is_primitive_and_get_value(choice_trie):
            if not choice_trie.is_primitive():
                raise RuntimError(f'choice_trie is not primitive: {choice_trie}')
            return choice_trie[()]

        def simulate(self, args):
            dist = dist_class(*args)
            value = dist.sample()
            lpdf = dist.log_prob(value).sum()
            return PyTorchDistTrace(self, value, lpdf)
    
        def generate(self, args, choice_trie):
            value = _check_is_primitive_and_get_value(choice_trie)
            dist = dist_class(*args)
            lpdf = dist.log_prob(value).sum()
            return (PyTorchDistTrace(self, value, lpdf), lpdf)

        # TODO this is an optimization; we would normally have to call choice_gradients or accum_param_grads
        def logpdf(self, args, value):
            dist = dist_class(*args)
            return dist.log_prob(value).sum()

    return gen_fn_class()

normal = torch_dist_to_gen_fn(torch.distributions.normal.Normal)
bernoulli = torch_dist_to_gen_fn(torch.distributions.bernoulli.Bernoulli)
