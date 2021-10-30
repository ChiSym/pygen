import ray
from ..dml.lang import DMLGenFn
#
#
# def configure_distributed_dml_runtime():
#     ray.init()


@ray.remote
class DistributedDMLGenFn(DMLGenFn):

    def __init__(self, inner):
        super().__init__(inner.p)
        self.p = inner.p
        self.torch_nn_module_children = inner.torch_nn_module_children
        self.torch_nn_module = inner.torch_nn_module

    @ray.method(num_returns=3)
    def propose(self, args):
        return super().propose(args)

    @ray.method(num_returns=2)
    def generate(self, args, constraints):
        return super().generate(args, constraints)

    # def generate_multi(self, args, constraints, n):
    #     traces_and_weights = []
    #     for i in range(n):
    #         future = self.generate.remote(args, constraints)
    #         traces_and_weights.append(future)
    #     return ray.get(traces_and_weights)