# MiniPyTorchGenDML

A minimal Gen dynamic modeling language (DML) implementation in PyTorch.

## Installation
Clone this repository:
```
git clone git@github.com:probcomp/mini-pygen-dml.git
```
Create a virtual environment that uses Python 3.6 or above, e.g.:
```
python3 -m venv .venv
```
Activate that virtual environment and upgrade pip
```
. .venv/bin/activate
python -m pip install -U pip
```
Install this package into the virtual environment
```
python -m pip install --editable .
```

## Run the tests
Install pytest:
```
python -m pip install pytest
```
Run pytest from the root directory of the repository:
```
python -m pytest tests
```

To generate a coverage report for the test suite, first install coverage:
```
python -m pip install coverage
```

Run the coverage script and inspect the results
```
./coverage.sh
xdg-open htmlcov/index.html
```

## Design

The modeling language implements the same GFI as Gen.jl but with a few tweaks (see limitations).

The implementation strategy closely mirrors that of the Gen.jl DML.

Users define a DML generative function by applying the `@gendml` decorator to a Python function.
Within the body of this Python function, the reserve keyword `gentrace` is used to (i) invoke other DML generative functions (no address argument permitted), (ii) sample random choices from probability distributions (third address argument required), and (iii) invoke `torch.nn.Module`s (no address argument permitted).

It is straightforward to invoke existing PyTorch modules (instances of `torch.nn.Module`) from a generative function, and to train the parameters of these modules using PyTorch's built-in optimizers (in concert with custom gradient accumulation schemes).
A DML generative function automatically constructs its own `torch.nn.Module` that has as children all PyTorch modules ever invoked during a traced execution of the generative function, that is accessible via the `get_torch_nn_module()` method.

Because the address namespace is flat, selections are simply Python built-in `set`s and choice maps are simply Python built-in `dict`s.

## Limitations

This implementation is designed primarily as a concrete reference point to aid in the design of a future version of Gen on top of PyTorch.

Some core features of Gen have not yet been added to this implementation:

- DML generative functions cannot invoke generative functions that are not implemented in the DML.

- DML generative functions can only invoke other DML generative functions via 'splicing' (i.e. without providing an extra address space). As a result there is no hierarchical address space.

- There is no support for change hints or incremental computation, and no combinators have been implemented.

- The 'regenerate' GFI method has not yet been implemented.

- There is no static modeling language (SML) implementation.

- Only fragments of the inference library have been implemented; but because the GFI was directly ported from Gen.jl it is straightforward to port the inference library code from Gen.jl (see e.g. `src/pygen/inflib/mcmc.py` for an example).

- The involution DSL has not been implemented (reference material for an implementation includes the PyTorch [implementation from the AIMCMC paper](https://github.com/probcomp/autoimcmc), the Gen.jl implementation, and [GenTraceKernelDSL.jl](https://github.com/probcomp/GenTraceKernelDSL.jl)).

Running on a GPU has not been tested, and no implementation effort has been spent on this.

## Next steps

Natural next steps are to:

- Implement benchmarks using this DML implementation (which has full functional capability of Gen.jl for the purposes of benchmarking performance of inference and learning programs, just lacking performance optimizations and engineering extensibility).

- Implement a GenList data type, following [GenCollections.jl](https://github.com/probcomp/GenCollections.jl)), and including combinators and incremental computation and automatic differentiation support, using [`pyrsistent`](https://github.com/tobgu/pyrsistent).

- Add support for hierarchical namespaces. This means that new types will likely be needed for selections and choice maps.

- Implement a builder for DAGs that mirrors the static IR builder in Gen.jl (but possibly with improvements to ergonomics to the interface)

- Implement the GFI for these DAGs

- Add more collections data types.
