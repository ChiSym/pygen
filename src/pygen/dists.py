from .gfi import Trace
import torch

class GenDistTrace(Trace):

    def __init__(self, value, lpdf):
        self.value = value
        self.lpdf = lpdf

    def get_score(self):
        return self.lpdf

    def get_retval(self):
        return self.value


class GenDist:
    pass

def torch_dist_to_gen_fn(dist_class):

    class gen_fn_class(GenDist):

        def simulate(self, args):
            dist = dist_class(*args)
            value = dist.sample()
            lpdf = dist.log_prob(value).sum()
            return GenDistTrace(value, lpdf)
    
        def generate(self, args, value):
            dist = dist_class(*args)
            lpdf = dist.log_prob(value).sum()
            return (GenDistTrace(value, lpdf), lpdf)

        def logpdf(self, args, value):
            dist = dist_class(*args)
            return dist.log_prob(value).sum()

    return gen_fn_class()

normal = torch_dist_to_gen_fn(torch.distributions.normal.Normal)
bernoulli = torch_dist_to_gen_fn(torch.distributions.bernoulli.Bernoulli)
