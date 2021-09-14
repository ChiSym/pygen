from pygen.dml.lang import gendml
from pygen.dists import bernoulli
from pygen.inflib.mcmc import mh_custom_proposal
from pygen import gentrace


@gendml
def model():
    z = gentrace(bernoulli, (0.5,), "z")
    assert z.size() == ()  # a scalar
    x_prob = (0.3 if z else 0.4)
    x = gentrace(bernoulli, (x_prob,), "x")


def z_conditional_prob(x):
    if x:
        z_false_prob = 0.5 * 0.4
        z_true_prob = 0.5 * 0.3
    else:
        z_false_prob = 0.5 * 0.6
        z_true_prob = 0.5 * 0.7
    return z_true_prob / (z_true_prob + z_false_prob)


@gendml
def proposal(trace):
    x = trace.get_choice_trie()["x"]
    gentrace(bernoulli, (z_conditional_prob(x),), "z")


iters = 50


def run_it():
    trace = model.simulate(())
    for i in range(iters):
        (trace, accepted) = mh_custom_proposal(trace, proposal, ())
        assert accepted
#
# import timeit
# number = 5
# total = timeit.timeit(run_it, number=number)
# rate = (number * iters) / total
# print(f"{rate} iters per second")

def test_always_accepts():
    pass
    run_it()
