from pygen.dml.lang import gendml
from pygen.dists import bernoulli, normal
from pygen.choice_address import addr
from pygen.choice_trie import MutableChoiceTrie
from pygen import gentrace
import torch
import torch.nn as nn


class ExampleTorchModule(nn.Module):

    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        return self.fc21(hidden)


network = ExampleTorchModule(2, 3, 4)


Z = addr('z')
X = addr('x')

@gendml
def f(mu):
    assert len(mu) == 2
    z = gentrace(normal, (mu, 1.0), Z)
    assert isinstance(z, torch.Tensor)
    assert z.size() == (2,)
    output = gentrace(network, (z,))
    assert isinstance(output, torch.Tensor)
    assert output.size() == (4,)
    x = gentrace(normal, (output, 1.0), X)
    assert isinstance(x, torch.Tensor)
    assert x.size() == (4,)
    return z


def get_expected_score(mu, z, x):
    with torch.inference_mode(True):
        score = (
            torch.distributions.normal.Normal(mu, 1.0).log_prob(z).sum() +
            torch.distributions.normal.Normal(network(z), 1.0).log_prob(x).sum())
    return score


def test_simulate():
    mu = torch.tensor([1.0, 2.0])
    trace = f.simulate((mu,))
    assert len(trace.get_choice_trie().asdict()) == 2
    z = trace.get_choice_trie()[Z]
    x = trace.get_choice_trie()[X]
    assert trace.get_gen_fn() == f
    assert trace.get_args() == (mu,)
    assert torch.allclose(trace.get_retval(), z)
    assert torch.allclose(trace.get_score(), get_expected_score(mu, z, x))


def test_generate():
    mu = torch.tensor([1.0, 2.0])
    x = torch.tensor([1.1, 1.2, 1.3, 1.4])
    trie = MutableChoiceTrie()
    trie[X] = x
    (trace, log_weight) = f.generate((mu,), trie)
    z = trace.get_choice_trie()[Z]
    assert len(trace.get_choice_trie().asdict()) == 2
    assert torch.allclose(trace.get_choice_trie()[X], x)
    assert trace.get_gen_fn() == f
    assert trace.get_args() == (mu,)
    assert torch.allclose(trace.get_retval(), z)
    assert torch.allclose(trace.get_score(), get_expected_score(mu, z, x))
    with torch.inference_mode(True):
        expected_log_weight = torch.distributions.normal.Normal(network(z), 1.0).log_prob(x).sum()
    assert torch.allclose(log_weight, expected_log_weight)


def test_update():
    mu = torch.tensor([1.0, 2.0])
    z = torch.tensor([1.1, 2.1])
    x = torch.tensor([1.1, 1.2, 1.3, 1.4])
    trie = MutableChoiceTrie()
    trie[Z] = z
    trie[X] = x
    (trace, _) = f.generate((mu,), trie)
    mu_new = torch.tensor([1.1, 2.1])
    z_new = torch.tensor([1.2, 2.2])
    trie = MutableChoiceTrie()
    trie[Z] = z_new
    (new_trace, log_weight, discard) = trace.update((mu_new,), trie)
    assert torch.allclose(log_weight, new_trace.get_score() - trace.get_score(), atol=1e-6)
    assert torch.allclose(new_trace.get_choice_trie()[Z], z_new)
    assert torch.allclose(new_trace.get_choice_trie()[X], x)
