# MiniPyTorchGenDML

A minimal Gen dynamic modeling language (DML) implementation in PyTorch.

## Installation
Clone this repository:
```
git clone git@github.com:probcomp/mini-pygen-dml.git
```
Create a virtual environment that uses Python 3.6 or above, e.g.:
```
virtualenv -p python3 .venv
```
Activate that virtual environment
```
source .venv/bin/activate
```
Install this package into the virtual environment
```
pip install --editable .
```

## Run the tests
Install pytest:
```
pip install pytest
```
Run pytest from the root directory of the repository:
```
pytest tests
```

## Design

The modeling language implements the same GFI as Gen.jl but with a few tweaks (see limitations).

The implementation strategy closely mirrors that of the Gen.jl DML.

It is straightforward to invoke existing PyTorch modules (instances of `torch.nn.Module`) from a generative function, and to train the parameters of these modules using PyTorch's built-in optimizers (in concert with custom gradient accumulation schemes).
A DML generative function automatically constructs its own `torch.nn.Module` that has as children all PyTorch modules ever invoked during a traced execution of the generative function, that is accessible via the `get_torch_nn_module()` method.

## Limitations

This implementation is designed primarily as a concrete reference point to aid in the design of a future version of Gen on top of PyTorch.

Some core features of Gen have not yet been added to this implementation:

- DML generative functions cannot invoke generative functions that are not implemented in the DML.

- DML generative functions can only invoke other DML generative functions via 'splicing' (i.e. without providing an extra address space). As a result there is no hierarchical address space.

- There is no support for change hints or incremental computation, and no combinators have not been implemented.

- The 'regenerate' GFI method has not yet been implemented

- There is no static modeling language (SML) implementation

- There is basically no inference library; but because the GFI was directly ported from Gen.jl it is straightforward to port the inference library code from Gen.jl (see e.g. `src/pygen/inflib/mcmc.py` for an example).

Running on a GPU has not been tested, and no implementation effort has been spent on this.
