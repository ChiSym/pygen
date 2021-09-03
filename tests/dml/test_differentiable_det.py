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

def check_param_gradients_unset_or_zero():
    for param in f.get_torch_nn_module().parameters():
        assert param.grad == None or torch.equal(param.grad, torch.zeros_like(param))

def check_param_gradients_nonzero():
    for param in f.get_torch_nn_module().parameters():
        assert param.grad != None
        assert not torch.equal(param.grad, torch.zeros_like(param))

def test_choice_gradients():
    mu = torch.tensor([1.0, 2.0])
    z = torch.tensor([1.1, 2.1])
    x = torch.tensor([1.1, 1.2, 1.3, 1.4])
    (trace, _) = f.generate((mu,), {"z" : z, "x" : x})
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

    # gradients wrt args and z
    (arg_grads, choice_values, choice_grads) = trace.choice_gradients(set(["z"]), retgrad)
    check_param_gradients_unset_or_zero()

    assert len(arg_grads) == 1
    mu_grad = arg_grads[0]
    assert mu_grad.size() == mu.size()
    assert torch.allclose(mu_grad, expected_mu_grad)

    assert len(choice_values) == 1
    z_choice = choice_values["z"]
    assert isinstance(z_choice, torch.Tensor)
    assert not z_choice.requires_grad
    assert torch.allclose(z_choice, z)

    assert len(choice_grads) == 1
    z_grad = choice_grads["z"]
    assert isinstance(z_choice, torch.Tensor)
    assert not z_choice.requires_grad
    assert z_grad.size() == z.size()
    assert not torch.allclose(z_grad, torch.zeros(z.size()))
    assert torch.allclose(z_grad, expected_z_grad)

    # gradients args, x, and z
    (arg_grads, choice_values, choice_grads) = trace.choice_gradients(set(["x", "z"]), retgrad)
    check_param_gradients_unset_or_zero()

    assert len(arg_grads) == 1
    mu_grad = arg_grads[0]
    assert mu_grad.size() == mu.size()
    assert torch.allclose(mu_grad, expected_mu_grad)

    assert len(choice_values) == 2
    z_choice = choice_values["z"]
    assert isinstance(z_choice, torch.Tensor)
    assert not z_choice.requires_grad
    assert torch.allclose(z_choice, z)
    x_choice = choice_values["x"]
    assert isinstance(x_choice, torch.Tensor)
    assert not x_choice.requires_grad
    assert torch.allclose(x_choice, x)

    assert len(choice_grads) == 2
    z_grad = choice_grads["z"]
    assert isinstance(z_choice, torch.Tensor)
    assert not z_choice.requires_grad
    assert z_grad.size() == z.size()
    assert torch.allclose(z_grad, expected_z_grad)
    x_grad = choice_grads["x"]
    assert isinstance(x_choice, torch.Tensor)
    assert not x_choice.requires_grad
    assert x_grad.size() == x.size()
    assert torch.allclose(x_grad, expected_x_grad)

def test_accumulate_param_gradients():
    mu = torch.tensor([1.0, 2.0])
    z = torch.tensor([1.1, 2.1])
    x = torch.tensor([1.1, 1.2, 1.3, 1.4])
    (trace, _) = f.generate((mu,), {"z" : z, "x" : x})
    retgrad = torch.tensor([0.3, 0.4])
    scale_factor = 1.123

    # compute expected_mu_grad
    mu_proxy = mu.detach().clone().requires_grad_(True)
    torch.autograd.backward(torch.distributions.normal.Normal(mu_proxy, 1.0).log_prob(z).sum())
    expected_mu_grad = mu_proxy.grad
    f.get_torch_nn_module().zero_grad()

    # compute expected gradients with respect to parameters
    network.requires_grad_(True)
    torch.autograd.backward(
        torch.distributions.normal.Normal(mu, 1.0).log_prob(z).sum() +
        torch.distributions.normal.Normal(network(z), 1.0).log_prob(x).sum())
    expected_param_grads = {}
    for (name, param) in f.get_torch_nn_module().named_parameters():
        expected_param_grads[name] = param.grad * scale_factor
    f.get_torch_nn_module().zero_grad()

    check_param_gradients_unset_or_zero()
    arg_grads = trace.accumulate_param_gradients(retgrad, scale_factor)
    check_param_gradients_nonzero()

    for (name, param) in f.get_torch_nn_module().named_parameters():
        assert torch.allclose(param.grad, expected_param_grads[name])

    assert len(arg_grads) == 1
    mu_grad = arg_grads[0]
    assert mu_grad.size() == mu.size()
    assert torch.allclose(mu_grad, expected_mu_grad)

    f.get_torch_nn_module().zero_grad()
