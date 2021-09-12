from pygen.choice_trie import MutableChoiceTrie
from pygen.choice_address import addr
from pygen.dml.lang import gendml
from pygen.dists import bernoulli
from pygen.inflib.importance import importance_sampling_custom_proposal
from pygen.inflib.importance import importance_resampling_custom_proposal
from pygen import gentrace
import torch

Z = addr('z')
X = addr('x')


@gendml
def model():
    z = gentrace(bernoulli, (0.1,), Z)
    assert z.size() == ()  # a scalar
    x_prob = (0.3 if z else 0.4)
    x = gentrace(bernoulli, (x_prob,), X)


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
    gentrace(bernoulli, (0.5,), Z)


@gendml
def exact_proposal(x):
    gentrace(bernoulli, (z_conditional_prob(x),), Z)


def test_marginal_likelihood_estimates():
    torch.manual_seed(0)
    x = torch.tensor(1.0)  # True
    lml_true = torch.log(torch.tensor(ground_truth_marginal_likelihood(x)))

    observations = MutableChoiceTrie()
    view = observations.flat_view()
    view[X] = x

    (_, lml_estimate) = importance_resampling_custom_proposal(
        model, (), observations, exact_proposal, (x,), 100, verbose=False)
    assert torch.isclose(lml_estimate, lml_true)

    (_, lml_estimate) = importance_resampling_custom_proposal(
        model, (), observations, uniform_proposal, (x,), 1000, verbose=False)
    assert torch.isclose(lml_estimate, lml_true, atol=1e-2)

    lml_true = torch.log(torch.tensor(ground_truth_marginal_likelihood(x)))
    (_, _, lml_estimate) = importance_sampling_custom_proposal(
        model, (), observations, exact_proposal, (x,), 100, verbose=False)
    assert torch.isclose(lml_estimate, lml_true)

    (_, _, lml_estimate) = importance_sampling_custom_proposal(
        model, (), observations, uniform_proposal, (x,), 1000, verbose=False)
    assert torch.isclose(lml_estimate, lml_true, atol=1e-2)
