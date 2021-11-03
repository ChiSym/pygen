import timeit
import torch
import torch.nn as nn
import pygen

from pygen.dml.lang import gendml, inline
from pygen.dists import bernoulli, normal
from pygen.choice_address import addr
from pygen.choice_trie import MutableChoiceTrie
from pygen.gfi import TorchModule

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

likelihood = TorchModule(LikelihoodModel(10, 3))

@gendml
def prior(zdim):
    z = normal(torch.zeros((zdim,)), 1.0) @ addr("z")
    return z


@gendml
def model(z_dim):
    z = prior(z_dim) @ inline
    pixel_probs = likelihood(z) @ inline
    binary_img = bernoulli(pixel_probs) @ addr("img")
    return binary_img

# Python call, does not produce return value like in Gen.jl
call = model(10)
assert isinstance(call, pygen.gfi.Call)

# simulate
trace = model.simulate((10,))
choices = trace.get_choice_trie()
retval = trace.get_retval()
score = trace.get_score()
print(f"score: {score}")
print(f"choices: {choices}")
print(f"retval: {retval}")

# generate
constraints = MutableChoiceTrie()
img_constraint = MutableChoiceTrie()
img_constraint.set_choice(torch.zeros((5,)))
constraints.set_subtrie(addr("img"), img_constraint)
(trace, log_weight) = model.generate((10,), constraints)
choices = trace.get_choice_trie()
print(f"choices: {choices}, log_weight: {log_weight}")

# update
update_choices = MutableChoiceTrie()
z_update = MutableChoiceTrie()
z_update.set_choice(torch.zeros((10,)))
update_choices.set_subtrie(addr("z"), z_update)
(new_trace, log_weight, discard) = trace.update((10,), update_choices)
new_choices = new_trace.get_choice_trie()
print(f"new_choices: {new_choices}, log_weight: {log_weight}, discard: {discard}")

n = 1000
elapsed = timeit.timeit(lambda: trace.update((10,), update_choices), number=n)
print(f"updates per second: {n/elapsed}")
print(elapsed)
