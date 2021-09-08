from pygen.dml.lang import gendml
from pygen.dists import bernoulli, normal
from pygen.inflib.importance import importance_sampling_custom_proposal
from pygen.inflib.importance import importance_resampling_custom_proposal
from pygen import gentrace
import torch
import timeit

@gendml
def model():
    z = gentrace(bernoulli, (0.1,), "z")
    assert z.size() == () # a scalar
    x_prob = (0.3 if z else 0.4)
    x = gentrace(bernoulli, (x_prob,), "x")

def ground_truth_marginal_likelihood(x):
    if x:
        z_false_prob = 0.9 * 0.4
        z_true_prob = 0.1 * 0.3
    else:
        z_false_prob = 0.9 * 0.6
        z_true_prob = 0.1 * 0.7
    return z_false_prob + z_true_prob

def z_conditional_prob(x):
    if x:
        z_false_prob = 0.9 * 0.4
        z_true_prob = 0.1 * 0.3
    else:
        z_false_prob = 0.9 * 0.6
        z_true_prob = 0.1 * 0.7
    return z_true_prob / (z_true_prob + z_false_prob)

@gendml
def uniform_proposal(x):
    gentrace(bernoulli, (0.5,), "z")

@gendml
def exact_proposal(x):
    gentrace(bernoulli, (z_conditional_prob(x),), "z")

def test_marginal_likelihood_estimates():
    torch.manual_seed(0)
    x = torch.tensor(1.0) # True

    lml_true = torch.log(torch.tensor(ground_truth_marginal_likelihood(x)))
    (_, lml_estimate) = importance_resampling_custom_proposal(
        model, (), {"x" : x}, exact_proposal, (x,), 100, verbose=False)
    assert torch.isclose(lml_estimate, lml_true)

    (_, lml_estimate) = importance_resampling_custom_proposal(
        model, (), {"x" : x}, uniform_proposal, (x,), 1000, verbose=False)
    assert torch.isclose(lml_estimate, lml_true, atol=1e-2)

    lml_true = torch.log(torch.tensor(ground_truth_marginal_likelihood(x)))
    (_, _, lml_estimate) = importance_sampling_custom_proposal(
        model, (), {"x" : x}, exact_proposal, (x,), 100, verbose=False)
    assert torch.isclose(lml_estimate, lml_true)

    (_, _, lml_estimate) = importance_sampling_custom_proposal(
        model, (), {"x" : x}, uniform_proposal, (x,), 1000, verbose=False)
    assert torch.isclose(lml_estimate, lml_true, atol=1e-2)
