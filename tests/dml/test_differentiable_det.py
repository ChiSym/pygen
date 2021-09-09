from pygen.dml.lang import gendml
from pygen.dists import bernoulli, normal
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

@gendml
def f(mu):
    assert len(mu) == 2
    z = gentrace(normal, (mu, 1.0), "z")
    assert isinstance(z, torch.Tensor)
    assert z.size() == (2,)
    output = gentrace(network, (z,))
    assert isinstance(output, torch.Tensor)
    assert output.size() == (4,)
    x = gentrace(normal, (output, 1.0), "x")
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
    assert len(trace.get_choices()) == 2
    z = trace.get_choices()["z"]
    x = trace.get_choices()["x"]
    assert trace.get_gen_fn() == f
    assert trace.get_args() == (mu,)
    assert torch.allclose(trace.get_retval(), z)
    assert torch.allclose(trace.get_score(), get_expected_score(mu, z, x))

def test_generate():
    mu = torch.tensor([1.0, 2.0])
    x = torch.tensor([1.1, 1.2, 1.3, 1.4])
    (trace, log_weight) = f.generate((mu,), {"x" : x})
    z = trace.get_choices()["z"]
    assert len(trace.get_choices()) == 2
    assert torch.allclose(trace.get_choices()["x"], x)
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
    (trace, _) = f.generate((mu,), {"z" : z, "x" : x})
    mu_new = torch.tensor([1.1, 2.1])
    z_new = torch.tensor([1.2, 2.2])
    (new_trace, log_weight, discard) = trace.update((mu_new,), {"z" : z_new})
    assert torch.allclose(log_weight, new_trace.get_score() - trace.get_score())
    assert torch.allclose(new_trace.get_choices()["z"], z_new)
    assert torch.allclose(new_trace.get_choices()["x"], x)
