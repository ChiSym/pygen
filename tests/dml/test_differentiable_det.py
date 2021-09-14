from pygen.dml.lang import gendml
from pygen.dists import bernoulli, normal
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


Z = 'z'
X = 'x'


@gendml
def g(mu):
    z = gentrace(normal, (mu, torch.tensor(1.0)), ())
    return z


@gendml
def f(mu):
    assert len(mu) == 2
    z = gentrace(g, (mu,), Z)
    assert isinstance(z, torch.Tensor)
    assert z.size() == (2,)
    output = gentrace(network, (z,))
    assert isinstance(output, torch.Tensor)
    assert output.size() == (4,)
    x = gentrace(normal, (output, torch.tensor(1.0)), X)
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


def check_param_gradients_unset_or_zero():
    for param in f.get_torch_nn_module().parameters():
        assert param.grad is None or torch.equal(param.grad, torch.zeros_like(param))


def test_choice_gradients():
    mu = torch.tensor([1.0, 2.0])
    z = torch.tensor([1.1, 2.1])
    x = torch.tensor([1.1, 1.2, 1.3, 1.4])
    trie = MutableChoiceTrie()
    trie[Z] = z
    trie[X] = x
    (trace, _) = f.generate((mu,), trie)
    retgrad = torch.tensor([0.3, 0.4])

    # compute expected_mu_grad
    mu_proxy = mu.detach().clone().requires_grad_(True)
    torch.autograd.backward(torch.distributions.normal.Normal(mu_proxy, 1.0).log_prob(z).sum())
    expected_mu_grad = mu_proxy.grad
    f.get_torch_nn_module().zero_grad()

    # compute expected_z_grad
    z_proxy = z.detach().clone().requires_grad_(True)
    torch.autograd.backward(
        [torch.distributions.normal.Normal(mu, 1.0).log_prob(z_proxy).sum() +
         torch.distributions.normal.Normal(network(z_proxy), 1.0).log_prob(x).sum(), z_proxy],
        grad_tensors=[torch.tensor(1.0), retgrad])
    expected_z_grad = z_proxy.grad
    f.get_torch_nn_module().zero_grad()

    # compute expected_x_grad
    x_proxy = x.detach().clone().requires_grad_(True)
    torch.autograd.backward(torch.distributions.normal.Normal(network(z), 1.0).log_prob(x_proxy).sum())
    expected_x_grad = x_proxy.grad
    f.get_torch_nn_module().zero_grad()

    # gradients wrt args only
    (arg_grads, choice_values, choice_grads) = trace.choice_gradients(None, retgrad)
    check_param_gradients_unset_or_zero()

    assert len(arg_grads) == 1
    mu_grad = arg_grads[0]
    assert mu_grad.size() == mu.size()
    assert torch.allclose(mu_grad, expected_mu_grad)

    assert choice_values is None
    assert choice_grads is None