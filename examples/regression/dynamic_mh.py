import numpy

# from pygen.dists import bernoulli
# from pygen.dists import normal
from pygen.dml.lang import gendml
from pygen.dml.lang import gentrace
from pygen.inflib.mcmc import mh_custom_proposal

from pygen.dists import GenDistTrace
from pygen.dists import GenDist
import scipy.stats

def scipy_dist_to_gen_fn_continuous(dist_class):
    class gen_fn_class(GenDist):
        def simulate(self, args):
            value = dist_class.rvs(*args)
            lpdf = dist_class.logpdf(value, *args)
            return GenDistTrace(value, lpdf)
        def generate(self, args, value):
            lpdf = dist_class.logpdf(value, *args)
            return (GenDistTrace(value, lpdf), lpdf)
        def logpdf(self, args, value):
            return dist_class.logpdf(value, *args)
    return gen_fn_class()

def scipy_dist_to_gen_fn_discrete(dist_class):
    class gen_fn_class(GenDist):
        def simulate(self, args):
            value = dist_class.rvs(*args)
            lpdf = dist_class.logpmf(value, *args)
            return GenDistTrace(value, lpdf)
        def generate(self, args, value):
            lpdf = dist_class.logpmf(value, *args)
            return (GenDistTrace(value, lpdf), lpdf)
        def logpdf(self, args, value):
            return dist_class.logpmf(value, *args)
    return gen_fn_class()

normal = scipy_dist_to_gen_fn_continuous(scipy.stats.norm)
bernoulli = scipy_dist_to_gen_fn_discrete(scipy.stats.bernoulli)

from dataset import make_data_set

@gendml
def datum(x, i, inlier_std, outlier_std, slope, intercept):
    is_outlier = gentrace(bernoulli, (.5,), ('data', i, 'z'))
    mu = x * slope + intercept
    std = inlier_std if is_outlier else outlier_std
    y = gentrace(normal, (mu, std), ('data', i, 'y'))
    return y

@gendml
def model(xs):
    log_inlier_std = gentrace(normal, (0, 2), 'log_inlier_std')
    log_outlier_std = gentrace(normal, (0, 2), 'log_outlier_std')
    inlier_std = numpy.exp(log_inlier_std)
    outlier_std = numpy.exp(log_outlier_std)
    slope = gentrace(normal, (0, 2), 'slope')
    intercept = gentrace(normal, (0, 2), 'intercept')
    ys = numpy.zeros(len(xs))
    for i, x in enumerate(xs):
        # TODO: No hierarchical addressing for subcall.
        ys[i] = gentrace(datum, (x, i, inlier_std, outlier_std, slope, intercept), None)
    return ys

@gendml
def proposal_gaussian_drift(trace, addr, std):
    prev = trace.get_choices()[addr]
    return gentrace(normal, (prev, std), addr)

@gendml
def proposal_flip(trace, addr):
    prev = trace.get_choices()[addr]
    weight = 0. if prev else 1.
    return gentrace(bernoulli, (weight,), addr)

def get_status(trace):
    choices = trace.get_choices()
    tokens = [
        ('score',        '%1.2f' % (trace.get_score()),),
        ('slope',        '%1.2f' % (choices['slope']),),
        ('intercept',    '%1.2f' % (choices['intercept']),),
        ('inlier_std',   '%1.2f' % (numpy.exp(choices['log_inlier_std'])),),
        ('outlier_std',  '%1.2f' % (numpy.exp(choices['log_outlier_std'])),),
    ]
    return ', '.join(': '.join(t) for t in tokens)

def run_inference(xs, ys, num_iters):
    observations = {('data', i, 'y'): y for (i, y) in enumerate(ys)}
    trace, score = model.generate((xs,), observations)
    print(get_status(trace))
    scores = [None] * num_iters
    for i in range(num_iters):
        # Steps on continuous parameters.
        for j in range(5):
            for addr in ['slope', 'intercept', 'log_inlier_std', 'log_outlier_std']:
                trace, _ = mh_custom_proposal(trace, proposal_gaussian_drift, (addr, .5))
        # Step on outlier indicators.
        for j in range(len(xs)):
            trace, _ = mh_custom_proposal(trace, proposal_flip, (('data', j, 'z'),))
        # Record results.
        scores[i] = trace.get_score()
        print(get_status(trace))
    return scores

(xs, ys) = make_data_set(200)
run_inference(xs, ys, 10)
