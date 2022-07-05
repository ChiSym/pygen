import time
import torch
import torch.nn as nn
import pygen
import matplotlib.pyplot as plt

from pygen.dists import normal, uniform
from pygen.dml.lang import gendml, inline
from pygen.choice_address import addr
from pygen.choice_trie import MutableChoiceTrie
from pygen.inflib.mcmc import mh_custom_proposal


@gendml
def normalModel():
    x = normal(0.0, 1.0) @ addr("x")
    return x


@gendml
def proposal(nowAt, d):
    choices = nowAt.get_choice_trie()
    current = choices[addr("x")]
    x = uniform(current - d, current + d) @ addr("x")
    return x


# Generate an initial trace.
trace = normalModel.simulate(())
M = 10
collect_elapsed = []

# Now, run MH inference with the custom proposal.
for iter in range(1, 3000):
    start = time.time()
    for i in range(2, 10):
        trace, _ = mh_custom_proposal(trace, proposal, (0.25,))
    end = time.time()
    elapsed = end - start
    print(f"iter: {iter} {elapsed}")
    collect_elapsed.append(elapsed)

fig, ax = plt.subplots()
ax.hist(collect_elapsed, 100)
ax.set(xlim=(0, 0.006))
plt.show()
