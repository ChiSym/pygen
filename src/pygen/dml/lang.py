from ..gfi import GenFn, Trace
from ..choice_address import ChoiceAddress, addr
from ..choice_trie import ChoiceTrie, MutableChoiceTrie
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


def _splice_dml_call(callee, args, gentrace):
    if isinstance(callee, DMLGenFn):
        p = _inject_variables({'gentrace': gentrace}, callee.p)
    else:
        raise RuntimeError('Address required when'
                           f' calling a non-DML generative function: {callee}')
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

        def gentrace(callee, callee_args, address=None):
            assert (address is None) or isinstance(address, ChoiceAddress)
            if isinstance(callee, GenFn):
                if address is None:
                    return _splice_dml_call(callee, callee_args, gentrace)
                else:
                    subtrace = callee.simulate(callee_args)
                    trace._record_subtrace(subtrace, address)
                    return subtrace.get_retval()
            elif isinstance(callee, torch.nn.Module):
                self._record_torch_nn_module(callee)
                return callee(*callee_args)
            else:
                raise RuntimeError('Unknown type of generative function:'
                                   f' {callee}')

        p = _inject_variables({'gentrace': gentrace}, self.p)
        with torch.inference_mode(mode=True):
            trace.retval = p(*args)
        return trace

    def generate(self, args, constraints):
        assert isinstance(constraints, ChoiceTrie)
        trace = DMLTrace(self, args)
        log_weight = torch.tensor(0.0)

        def gentrace(callee, callee_args, address=None):
            assert (address is None) or isinstance(address, ChoiceAddress)
            if isinstance(callee, GenFn):
                if address is None:
                    return _splice_dml_call(callee, callee_args, gentrace)
                sub_constraints = constraints.get_subtrie(address, strict=False)
                (subtrace, log_weight_increment) = callee.generate(callee_args, sub_constraints)
                nonlocal log_weight
                log_weight += log_weight_increment
                trace._record_subtrace(subtrace, address)
                return subtrace.get_retval()
            if isinstance(callee, torch.nn.Module):
                self._record_torch_nn_module(callee)
                return callee(*callee_args)
            raise RuntimeError(f'Unknown type of generative function: {callee}')

        p = _inject_variables({'gentrace': gentrace}, self.p)
        with torch.inference_mode(mode=True):
            trace.retval = p(*args)
        return (trace, log_weight)

    def __call__(self, *args):
        trace = self.simulate(args)
        return trace.get_retval()


def torch_autograd_function_from_trace(trace):

    class AutogradFunctionForTrace(torch.autograd.Function):

        @staticmethod
        def forward(ctx, *args):
            return trace.get_score(), trace.get_retval()

        @staticmethod
        def backward(ctx, score_increment_grad, retval_grad):
            assert isinstance(score_increment_grad, torch.Tensor)
            assert isinstance(retval_grad, torch.Tensor)
            assert not score_increment_grad.requires_grad
            assert not retval_grad.requires_grad
            # NOTE: score_increment_grad is a dummy value
            arg_grads, choice_dict, grad_dict = trace.choice_gradients(None, retval_grad)
            assert isinstance(arg_grads, tuple)
            for grad in arg_grads:
                assert isinstance(grad, torch.Tensor)
                assert not grad.requires_grad
            assert choice_dict is None
            assert grad_dict is None
            return arg_grads

    return AutogradFunctionForTrace.apply


class DMLTrace(Trace):

    def __init__(self, gen_fn, args):
        self.gen_fn = gen_fn
        self.args = args
        self.score = torch.tensor(0.0)
        self.retval = None
        self.empty_address_subtrace = None
        self.subtraces_trie = {}

    @staticmethod
    def _record_subtrace_in_subtraces_trie(subtraces_trie, address, subtrace, full_address):
        assert isinstance(address, ChoiceAddress) and address
        first = address.first()
        rest = address.rest()
        if not rest:
            if first in subtraces_trie:
                raise RuntimeError(f'Address {full_address} already visited; cannot sample a choice at it')
            subtraces_trie[first] = subtrace
        else:
            if first not in subtraces_trie:
                subtraces_trie[first] = {}
            DMLTrace._record_subtrace_in_subtraces_trie(subtraces_trie[first], rest, subtrace, full_address)

    def _record_subtrace(self, subtrace, addr):
        assert isinstance(subtrace, Trace)
        assert isinstance(addr, ChoiceAddress)
        value = subtrace.get_retval()
        assert not value.requires_grad
        if not addr:
            if (self.empty_address_subtrace is not None) or self.subtraces_trie:
                raise RuntimeError('the empty address may be visited at most once, and must be the only'
                                   f' address visited, but address {next(iter(self.subtraces_trie))}'
                                   ' was also visited')
            self.empty_address_subtrace = subtrace
        else:
            if self.empty_address_subtrace is not None:
                raise RuntimeError('the empty address may be visited at most once,'
                                   f' and must be the only address visited, but address {addr}'
                                   'was also visited')
            DMLTrace._record_subtrace_in_subtraces_trie(self.subtraces_trie, addr, subtrace, addr)
        score_increment = subtrace.get_score()
        self.score += score_increment

    def get_gen_fn(self):
        return self.gen_fn

    def get_args(self):
        return self.args

    def get_score(self):
        return self.score

    def get_retval(self):
        return self.retval

    @staticmethod
    def _to_choice_trie(subtraces_trie):
        # subtraces_trie is a trie where leaves are Traces
        assert isinstance(subtraces_trie, dict)
        trie = MutableChoiceTrie()
        for (k, v) in subtraces_trie.items():
            if isinstance(v, Trace):
                trie.set_subtrie(addr(k), v.get_choice_trie())
            else:
                assert isinstance(v, dict)
                trie.set_subtrie(addr(k), DMLTrace._to_choice_trie(v))
        return trie

    def get_choice_trie(self):
        if self.empty_address_subtrace is not None:
            trie = MutableChoiceTrie()
            trie.set_choice(self.empty_address_subtrace.get_retval())
            return trie
        else:
            return DMLTrace._to_choice_trie(self.subtraces_trie)

    @staticmethod
    def _deleted_subtraces_score(prev_subtraces_trie, new_subtraces_trie):
        score = torch.tensor(0.0, requires_grad=False)
        for (k, v) in prev_subtraces_trie.items():
            if isinstance(v, Trace):
                if k not in new_subtraces_trie:
                    score += v.get_score()
            else:
                assert isinstance(v, dict)
                if k in new_subtraces_trie:
                    score += DMLTrace._deleted_subtraces_score(v, new_subtraces_trie[v])
                else:
                    score += DMLTrace._deleted_subtraces_score(v, {})
        return score

    @staticmethod
    def _add_unvisited_to_discard(discard, prev_subtraces_trie, new_subtraces_trie):
        for (k, v) in prev_subtraces_trie.items():
            if isinstance(v, Trace):
                if k not in new_subtraces_trie:
                    discard.set_subtrie(addr(k),  v.get_choice_trie())
            else:
                assert isinstance(v, dict)
                if k in new_subtraces_trie:
                    sub_discard = discard.get_subtrie(addr(k), strict=False)
                else:
                    sub_discard = MutableChoiceTrie()
                DMLTrace._add_unvisited_to_discard(sub_discard, v, new_subtraces_trie[v])
                discard.set_subtrie(addr(k), sub_discard)

    def _get_subtrace(self, subtraces_trie, address):
        assert isinstance(address, ChoiceAddress)
        # empty address.
        if not address:
            return self.empty_address_subtrace
        # non-empty address
        first = address.first()
        rest = address.rest()
        if first not in subtraces_trie:
            return None
        if rest:
            trie = subtraces_trie[first]
            if not isinstance(trie, dict):
                # there was previously a subtrace at this address, but now there is a subtrace at a
                # an extension of this address
                return None
            return self._get_subtrace(subtraces_trie[first], rest)
        subtrace = subtraces_trie[first]
        if not isinstance(subtrace, Trace):
            # there was previously a subtrace at an extension of this address, but now there this a
            # subtrace at this address
            return None
        return subtrace

    def update(self, args, constraints):
        new_trace = DMLTrace(self.get_gen_fn(), args)
        discard = MutableChoiceTrie()
        log_weight = torch.tensor(0.0, requires_grad=False)

        def gentrace(callee, callee_args, address=None):
            assert (address is None) or isinstance(address, ChoiceAddress)
            if isinstance(callee, GenFn):
                if address is None:
                    return _splice_dml_call(callee, callee_args, gentrace)
                nonlocal log_weight
                prev_subtrace = self._get_subtrace(self.subtraces_trie, address)
                if prev_subtrace:
                    if prev_subtrace.get_gen_fn() != callee:
                        raise RuntimeError(f'Generative function at address {address}'
                                           'changed from {prev_callee} to {callee}')
                    (subtrace, log_weight_increment, sub_discard) = prev_subtrace.update(
                        callee_args, constraints.get_subtrie(address, strict=False))
                    if sub_discard:
                        discard.set_subtrie(address, sub_discard)
                else:
                    (subtrace, log_weight_increment) = callee.generate(
                        callee_args, constraints.get_subtrie(address, strict=False))
                log_weight += log_weight_increment
                new_trace._record_subtrace(subtrace, address)
                return subtrace.get_retval()
            if isinstance(callee, torch.nn.Module):
                self.get_gen_fn()._record_torch_nn_module(callee)
                return callee(*callee_args)
            raise RuntimeError('Unknown type of generative function:'
                                   f' {callee}')

        p = _inject_variables({'gentrace': gentrace}, self.get_gen_fn().p)
        with torch.inference_mode(mode=True):
            new_trace.retval = p(*args)

        log_weight -= DMLTrace._deleted_subtraces_score(self.subtraces_trie, new_trace.subtraces_trie)
        self._add_unvisited_to_discard(discard, self.subtraces_trie, new_trace.subtraces_trie)

        return (new_trace, log_weight, discard)

    def regenerate(self, args, constraints):
        raise NotImplementedError()

    def choice_gradients(self, selection, retval_grad):
        if selection is not None:
            raise NotImplementedError() # TODO add support for gradients wrt choices.
        with torch.inference_mode(mode=False):
            score = torch.tensor(0.0, requires_grad=False)

            def gentrace(callee, callee_args, address=None):
                if isinstance(callee, GenFn):
                    if address is None:
                        return _splice_dml_call(callee, callee_args, gentrace)
                    for arg in callee_args:
                        if not isinstance(arg, torch.Tensor):
                            raise NotImplementedError('Only Tensor arguments are currently supported')
                    with torch.inference_mode(mode=True):
                        prev_subtrace = self._get_subtrace(self.subtraces_trie, address)
                        torch_autograd_function = torch_autograd_function_from_trace(prev_subtrace)
                    (score_increment, callee_retval) = torch_autograd_function(*callee_args)
                    nonlocal score
                    score += score_increment
                    if not isinstance(callee_retval, torch.Tensor):
                        raise NotImplementedError('Only a single Tensor return value is currently')
                    return callee_retval
                if isinstance(callee, torch.nn.Module):
                    for param in callee.parameters():
                        param.requires_grad_(False)
                    return callee(*callee_args)
                raise RuntimeError('Unknown type of generative function: {callee}')

            p = _inject_variables({'gentrace' : gentrace}, self.gen_fn.p)
            args_tracked = tuple(
                arg.detach().clone().requires_grad_(True) if isinstance(arg, torch.Tensor) else arg
                for arg in self.get_args())
            retval = p(*args_tracked)

            arg_grads = torch.autograd.grad(
                (score, retval), args_tracked,
                grad_outputs=(torch.tensor(1.0), retval_grad))

        return arg_grads, None, None

    def accumulate_param_gradients(self, retgrad, scale_factor):
        raise NotImplementedError()


def gendml(p):
    return DMLGenFn(p)