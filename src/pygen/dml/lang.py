from ..gfi import GenFn, Trace
from ..dists import GenDist, GenDistTrace
from functools import wraps
import torch

# inject variables into a function's scope:
def _inject_variables(context, func):
    @wraps(func)
    def new_func(*args, **kwargs):
        func_globals = func.__globals__
        saved_values = func_globals.copy()
        func_globals.update(context)
        try:
            result = func(*args, **kwargs)
        finally:
            for (var, val) in context.items():
                if var in saved_values:
                    func_globals.update({var: saved_values[var]})
                else:
                    del func_globals[var]
        return result
    return new_func
def splice_dml_call(gen_fn, args, addr, gentrace):
    if addr is not None:
        raise RuntimeError("Address must not be provided for a DML call, got: {addr}")
    p = _inject_variables({"gentrace" : gentrace}, gen_fn.p)
    return p(*args)

class DMLGenFn(GenFn):

    def __init__(self, p):
        self.p = p
        self.torch_nn_module_children = set()
        self.torch_nn_module = torch.nn.Module()

    def _record_torch_nn_module(self, module):
        if not module in self.torch_nn_module_children:
            name = str(len(self.torch_nn_module_children))
            self.torch_nn_module.add_module(name, module)

    def get_torch_nn_module(self):
        return self.torch_nn_module

    def simulate(self, args):
        trace = DMLTrace(self, args)

        def gentrace(callee, args, addr=None):
            if isinstance(callee, DMLGenFn):
                # recursive calls to this 'gentrace'
                return splice_dml_call(callee, args, addr, gentrace)
            elif isinstance(callee, GenDist):
                if addr is None:
                    raise RuntimeError("Address must be provided for a GenDist call")
                subtrace = callee.simulate(args)
                trace._record_choice(subtrace, addr)
                return subtrace.get_retval()
            elif isinstance(callee, torch.nn.Module):
                self._record_torch_nn_module(callee)
                return callee(*args)
            else:
                raise RuntimeError("Unknown type of generative function: {callee}")

        p = _inject_variables({"gentrace" : gentrace}, self.p)
        with torch.inference_mode(mode=True):
            trace.retval = p(*args)
        return trace

    def generate(self, args, constraints):
        trace = DMLTrace(self, args)
        log_weight = torch.tensor(0.0, requires_grad=False)

        def gentrace(callee, args, addr=None):
            if isinstance(callee, DMLGenFn):
                # recursive calls to this 'gentrace'
                return splice_dml_call(callee, args, addr, gentrace)
            elif isinstance(callee, GenDist):
                if addr is None:
                    raise RuntimeError("Address must be provided for a GenDist call")
                if addr in constraints:
                    (subtrace, log_weight_incr) = callee.generate(args, constraints[addr])
                    nonlocal log_weight
                    log_weight += log_weight_incr
                else:
                    subtrace = callee.simulate(args)
                trace._record_choice(subtrace, addr)
                return subtrace.get_retval()
            elif isinstance(callee, torch.nn.Module):
                self._record_torch_nn_module(callee)
                return callee(*args)
            else:
                raise RuntimeError("Unknown type of generative function: {callee}")

        p = _inject_variables({"gentrace" : gentrace}, self.p)
        with torch.inference_mode(mode=True):
            trace.retval = p(*args)
        return (trace, log_weight)

    def __call__(self, *args):
        trace = self.simulate(args)
        return trace.get_retval()


class DMLTrace(Trace):

    def __init__(self, gen_fn, args):
        self.gen_fn = gen_fn
        self.args = args
        self.score = torch.tensor(0.0, requires_grad=False)
        self.retval = None
        self.choices = {}
        self.choice_scores = {}

    def _record_choice(self, subtrace, addr):
        if addr in self.choices:
            raise RuntimeError(f'address {addr} is already visited')
        value = subtrace.get_retval()
        assert not value.requires_grad
        self.choices[addr] = value
        score_increment = subtrace.get_score()
        self.choice_scores[addr] = score_increment
        self.score += score_increment

    def get_gen_fn(self):
        return self.gen_fn

    def get_args(self):
        return self.args

    def get_score(self):
        return self.score

    def get_retval(self):
        return self.retval

    def get_choices(self):
        return self.choices

    def update(self, args, constraints):
        new_trace = DMLTrace(self.get_gen_fn(), args)
        discard = {}
        log_weight = torch.tensor(0.0, requires_grad=False)

        def gentrace(callee, args, addr=None):
            if isinstance(callee, DMLGenFn):
                # recursive calls to this 'gentrace'
                return splice_dml_call(callee, args, addr, gentrace)
            elif isinstance(callee, GenDist):
                if addr is None:
                    raise RuntimeError("Address must be provided for a GenDist call")

                has_previous = (addr in self.choices)
                constrained = (addr in constraints)

                if has_previous:
                    prev_value = self.choices[addr]
                    prev_score = self.choice_scores[addr]

                if constrained and has_previous:
                    discard[addr] = prev_value

                if constrained:
                    new_value = constraints[addr]
                elif has_previous:
                    new_value = prev_value
                else:
                    subtrace = callee.simulate(args)
                    new_value = subtrace.get_value()

                new_score = callee.logpdf(args, new_value)
                nonlocal log_weight
                log_weight += new_score
                new_trace._record_choice(GenDistTrace(new_value, new_score), addr)
                return new_value
            elif isinstance(callee, torch.nn.Module):
                self.get_gen_fn()._record_torch_nn_module(callee)
                return callee(*args)
            else:
                raise RuntimeError("Unknown type of generative function: {callee}")

        p = _inject_variables({"gentrace" : gentrace}, self.get_gen_fn().p)
        with torch.inference_mode(mode=True):
            new_trace.retval = p(*args)

        log_weight -= self.get_score()
        for (addr, value) in self.get_choices().items():
            if not addr in new_trace.choices:
                discard[addr] = value

        return (new_trace, log_weight, discard)


    def regenerate(self, args, constraints):
        raise NotImplementedError()

    def choice_gradients(self, selection, retval_grad):
        with torch.inference_mode(mode=False):
            score = torch.tensor(0.0, requires_grad=False)
            choice_dict = {}

            def gentrace(callee, args, addr=None):
                nonlocal score
                if isinstance(callee, DMLGenFn):
                    # recursive calls to this 'gentrace'
                    return splice_dml_call(callee, args, addr, gentrace)
                elif isinstance(callee, GenDist):
                    if addr is None:
                        raise RuntimeError("Address must be provided for non-DML call")
                    value = self.choices[addr]
                    assert not value.requires_grad
                    if addr in selection:
                        leaf_value = value.detach().clone().requires_grad_(True)
                        choice_dict[addr] = leaf_value
                        score += callee.logpdf(args, leaf_value)
                        return leaf_value
                    else:
                        score += callee.logpdf(args, value)
                        return value
                elif isinstance(callee, torch.nn.Module):
                    for param in callee.parameters():
                        param.requires_grad_(False)
                    return callee(*args)
                else:
                    raise RuntimeError("Unknown type of generative function: {callee}")

            p = _inject_variables({"gentrace" : gentrace}, self.gen_fn.p)
            args_tracked = tuple(
                arg.detach().clone().requires_grad_(True) if isinstance(arg, torch.Tensor) else arg
                for arg in self.get_args())
            retval = p(*args_tracked)
            req = retval.requires_grad
            grad_fn = retval.grad_fn

            if retval.requires_grad:
                retval.backward(gradient=retval_grad, retain_graph=False)
            score.backward(retain_graph=True)

            arg_grads = tuple(
                arg.grad if isinstance(arg, torch.Tensor) else None
                for arg in args_tracked)

            grad_dict = {}
            for (addr, leaf_value) in choice_dict.items():
                grad_dict[addr] = leaf_value.grad.detach()
                leaf_value.requires_grad_(False)

        return (arg_grads, choice_dict, grad_dict)

    def accumulate_param_gradients(self, retgrad, scale_factor):
        with torch.inference_mode(mode=False):
            score = torch.tensor(0.0, requires_grad=False)
            choice_dict = {}

            def gentrace(callee, args, addr=None):
                nonlocal score
                if isinstance(callee, DMLGenFn):
                    # recursive calls to this 'gentrace'
                    return splice_dml_call(callee, args, addr, gentrace)
                elif isinstance(callee, GenDist):
                    if addr is None:
                        raise RuntimeError("Address must be provided for non-DML call")
                    value = self.choices[addr].detach().clone().requires_grad_(False)
                    assert not value.requires_grad
                    score += callee.logpdf(args, value)
                    return value
                elif isinstance(callee, torch.nn.Module):
                    for param in callee.parameters():
                        param.requires_grad_(True)
                    return callee(*args)
                else:
                    raise RuntimeError("Unknown type of generative function: {callee}")

            p = _inject_variables({"gentrace" : gentrace}, self.gen_fn.p)
            args_tracked = tuple(
                arg.detach().requires_grad_(True) if isinstance(arg, torch.Tensor) else arg
                for arg in self.get_args())
            retval = p(*args_tracked)
            req = retval.requires_grad
            grad_fn = retval.grad_fn

            # multiply the existing gradient by 1/scale_factor
            for param in self.get_gen_fn().get_torch_nn_module().parameters():
                if param.grad is not None:
                    param.grad.mul_(1.0 / scale_factor)

            # then accumulate gradient with scale factor of 1
            if retval.requires_grad:
                retval.backward(gradient=retval_grad, retain_graph=False)
            score.backward(retain_graph=True)

            # then multiply the total gradient by scale_factor
            for param in self.get_gen_fn().get_torch_nn_module().parameters():
                param.grad.mul_(scale_factor)

            arg_grads = tuple(
                arg.grad if isinstance(arg, torch.Tensor) else None
                for arg in args_tracked)

        return arg_grads


def gendml(p):
    return DMLGenFn(p)
