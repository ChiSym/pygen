from .gfi import Trace, GenFn
from .address import addr, ChoiceAddress
from .choice_trie import ChoiceTrie, MutableChoiceTrie
import torch

class TorchDist(GenFn):
    pass

class TorchDistTrace(Trace):

    def __init__(self, gen_fn, args, value, lpdf):
        assert isinstance(gen_fn, TorchDist)
        self.gen_fn
        self.args = args
        self.value = value
        self.lpdf = lpdf

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
        trie[addr()] = self.value
        return trie

    def update(self, args, choice_trie):
        assert isinstance(choice_trie, ChoiceTrie)
        discard = MutableChoiceTrie()
        prev_value = self.value
        if not choice_trie:
            # choice_trie is empty
            value = prev_value
        else:
            # choice_trie is not empty
            value = _check_is_primitive_and_get_value(choice_trie)
            discard[addr()] = prev_value
        new_dist = self.gen_fn.get_dist_class()(*args)
        new_lpdf = new_dist.log_prob(value).sum()
        prev_lpdf = self.lpdf
        log_weight = new_lpdf - prev_lpdf
        new_trace = TorchDistTrace(self.gen_fn, args, value, new_lpdf)
        return (new_trace, log_weight, discard)

    def regenerate(self, args, selection):
        raise NotImplementedError()

    # NOTE: as an optimization, these methods currently use the 'logpdf' method
    # (see below) instead..

    def choice_gradients(self, selection, retgrad):
        raise NotImplementedError()

    def accum_param_grads(self, retgrad, scale_factor):
        raise NotImplementedError()


def torch_dist_to_gen_fn(dist_class):

    class gen_fn_class(TorchDist):

        def __init__(self):
            self.dist_class = dist_class

        def get_dist_class(self):
            return self.dist_class

        @staticmethod
        def _check_is_primitive_and_get_value(choice_trie):
            if not choice_trie.is_primitive():
                raise RuntimError(f'choice_trie is not primitive: {choice_trie}')
            return choice_trie[addr()]

        def simulate(self, args):
            dist = dist_class(*args)
            value = dist.sample()
            lpdf = dist.log_prob(value).sum()
            return TorchDistTrace(self, value, lpdf)
    
        def generate(self, args, choice_trie):
            value = _check_is_primitive_and_get_value(choice_trie)
            dist = dist_class(*args)
            lpdf = dist.log_prob(value).sum()
            return (TorchDistTrace(self, value, lpdf), lpdf)

        # TODO this is an optimization; we would normally have to call choice_gradients or accum_param_grads
        def logpdf(self, args, value):
            dist = dist_class(*args)
            return dist.log_prob(value).sum()

    return gen_fn_class()

normal = torch_dist_to_gen_fn(torch.distributions.normal.Normal)
bernoulli = torch_dist_to_gen_fn(torch.distributions.bernoulli.Bernoulli)
