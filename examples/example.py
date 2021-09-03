from pygen.dml.lang import gendml
from pygen.gfi import Trace, GenFn
from pygen.dists import bernoulli, normal
import torch
import torch.nn as nn
import timeit

# example of a torch.nn.Module:

class LikelihoodModel(nn.Module):

    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 5)
        self.softplus = nn.Softplus()
        self.z_dim = z_dim

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        pixel_probs = torch.sigmoid(self.fc21(hidden))
        return pixel_probs

@gendml
def prior(zdim):
    z = gentrace(normal, (torch.zeros((zdim,)), 1.0), "z")
    return z

likelihood = LikelihoodModel(10, 3)

@gendml
def model(z_dim):
    z = gentrace(prior, (z_dim,))
    pixel_probs = gentrace(likelihood, (z,))
    binary_img = gentrace(bernoulli, (pixel_probs,), "img")
    return binary_img
    
# simulate
trace = model.simulate((10,))
choices = trace.get_choices()
retval = trace.get_retval()
score = trace.get_score()
print(f"score: {score}")
print(f"choices: {choices}")
print(f"retval: {retval}")

# generate
(trace, log_weight) = model.generate((10,), {"img" : torch.zeros((5,))})
choices = trace.get_choices()
print(f"choices: {choices}, log_weight: {log_weight}")

# update
(new_trace, log_weight, discard) = trace.update((10,), {"z" : torch.zeros((10,))})
new_choices = new_trace.get_choices()
print(f"new_choices: {new_choices}, log_weight: {log_weight}, discard: {discard}")

# choice_gradients
(arg_grads, choice_values, choice_grads) = trace.choice_gradients(set(["z", "img"]), torch.zeros((5,)))
print(f"arg_grads: {arg_grads}, choice_values: {choice_values}, choice_grads: {choice_grads}")

module = model.get_torch_nn_module()

for param in module.parameters():
    assert param.grad == None or torch.equal(param.grad, torch.zeros_like(param))

# accumulate param gradients
arg_grads = trace.accumulate_param_gradients(torch.zeros((5,)), 1.0)
print(f"arg_grads: {arg_grads}")

for param in module.parameters():
    assert param.grad != None and not torch.equal(param.grad, torch.zeros_like(param))

module.zero_grad()

for param in module.parameters():
    assert param.grad != None and torch.equal(param.grad, torch.zeros_like(param))

n = 1000
elapsed = timeit.timeit(lambda: trace.update((10,), {"z" : torch.zeros((10,))}), number=n)
print(f"updates per second: {n/elapsed}")
print(elapsed)
