# Progress

#### Milestone 0

[X] fully-functional GFI implementation for DML with splicing and distribution calls only (no hierarchical addresses)

#### Milestone 1 

[X] choice tries are implemented and tested

[X] simulate, generate, and update are implemented and tested, including for these 'non-spliced' generative functions

[  ] there is some example of an 'external' (non-DML) generative function that gets invoked (perhaps Feras' multivariate-normal generative function)

[X] choice_gradients and accumulate_param_gradients raise NotImplementedErrors

#### Milestone 2

[X] make AD work when arguments and return values of non-spliced generative functions are either Tensors OR Python tuples, lists, and dicts

[X] accumulate_param_grads implemented and tested for hierarchical address version

[~] hierarchical selections implemented and tested

[ ] choice_gradients implemented and tested for the hierarchical address case

(this gives a fully-functional GFI with hierarchical addresses)

#### Milestone 3

[  ] change hints and incremental computation (update and regenerate), building on `pyrsistent`

[  ] implement unfold combinator and test with particle filtering in HMM

[  ] implement regenerate

#### Milestone 4

[  ] implement SML backend (static Gen IR)

[  ] implement SML front-end

#### Milestone 5 

[  ] C++ version :)

# pygen-experimental

An experimental port of Gen to Python, PyTorch, and maybe, C++.

## Installation
Clone this repository:
```
git clone git@github.com:probcomp/pygen-experimental.git
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

### DML

The modeling language in `src/pygen/dml/` implements the same GFI as Gen.jl but with a few tweaks (see limitations).
The implementation strategy closely mirrors that of the Gen.jl DML.

Users define a DML generative function by applying the `@gendml` decorator to a Python function.
Within the body of this Python function, you can
(i) invoke other generative functions, using the syntax `<gen_fn>(<args...>) @ <addr>`, and
(ii) invoke `torch.nn.Module`s (first, wrap the `torch.nn.Module` instance in a `pygen.gfi.TorchModule` instance e.g. `f`, and then use `f(<args...>) @ inline`).

It is straightforward to invoke existing PyTorch modules (instances of `torch.nn.Module`) from a generative function, and to train the parameters of these modules using PyTorch's built-in optimizers (in concert with custom gradient accumulation schemes).
A DML generative function automatically constructs its own `torch.nn.Module` that has as children all PyTorch modules ever invoked during a traced execution of the generative function, that is accessible via the `get_torch_nn_module()` method.

The address namespace is hierarchical.
You can invoke another DML generative function using the special `pygen.dml.lang.inline` constant as the address after `@` to 'inline' the trace and not introduce a new address namespace for the call.

### Other

Note that unlike in Gen.jl's DML, primitive distributions are also generative functions.

Currently, the only implementation of the choice trie interface is `MutableChoiceTrie`.

Automatic differentiation works through arguments and return-values of generative functions that are `torch.Tensor`s, Python `list`s, Python `dict`s, Python `tuple`s, and compositions of these, as well as user-registered compound data types.

## Limitations

This implementation is designed primarily as a concrete reference point to aid in the design of a future version of Gen on top of PyTorch.

Some core features of Gen have not yet been added to this implementation:

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

- Implement selections.

- Implement a builder for DAGs that mirrors the static IR builder in Gen.jl (but possibly with improvements to ergonomics to the interface)

- Implement the GFI for these DAGs

- Add more collections data types.

More speculative next steps:

- Implement DML and SML languages, embedded in C++, building on PyTorch's C++ API, called LibTorch. Early experiments by Marco suggest that high-performance multi-threaded Monte Carlo inference on a CPU may work well. Implement a basic learning and inference library in C++.

- Link up the Python implementation with the C++ implementation. TorchScript (via `torch.jit.script`) is a good candidate vehicle for getting user modeling code (at least deterministic fragments, especially for neural networks) imported to be callable by C++ generative functions. A C++ SML (and combinators) implementation (as a builder) can be exposed to Python users. The C++ inference library could also be exposed to Python.

- If a user really wants to just implement models in Python, but the learning and inference library is in C++, an RPC/IPC mechanism could be used to invoke Python generative functions (including DML functions that use arbitrary Python control flow) from the C++ inference library and C++ generative functions.

- Some careful thinking about how to effectively use GPUs at training time is needed. The current trace data type approach does not provide special support for vectorization of models, and vectorization may be fundamentally less powerful for models with latent structure.
