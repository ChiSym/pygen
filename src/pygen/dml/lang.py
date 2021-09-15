from ..gfi import GenFn, Trace, Unknown
from ..choice_address import ChoiceAddress
from ..choice_trie import ChoiceTrie, MutableChoiceTrie
import pygen.gradients as gradients

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
        raise RuntimeError('Address required when calling a non-DML '
            f'generative function: {callee}')
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
                raise RuntimeError(f'Unknown type of generative function: {callee}')

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
        def forward(ctx, *args_unrolled):
            retval = trace.get_retval()
            retval_unrolled = gradients.unroll_torch_tensors(retval, detach=True)
            return (trace.get_score(), *retval_unrolled)

        @staticmethod
        def backward(ctx, score_increment_grad, *retval_grad_unrolled):
            assert isinstance(score_increment_grad, torch.Tensor)
            retval_grad = gradients.roll_torch_tensors(trace.get_retval(), retval_grad_unrolled)
            assert not score_increment_grad.requires_grad
            for t in retval_grad_unrolled:
                assert not t.requires_grad
            # NOTE: score_increment_grad is a dummy value
            arg_grads, choice_dict, grad_dict = trace.choice_gradients(None, retval_grad)
            assert isinstance(arg_grads, tuple)
            arg_grads_unrolled = gradients.unroll_torch_tensors(arg_grads)
            for t in arg_grads_unrolled:
                assert isinstance(t, torch.Tensor)
                assert not t.requires_grad
            assert choice_dict is None
            assert grad_dict is None
            return arg_grads_unrolled

    return AutogradFunctionForTrace.apply


class DMLTrace(Trace):

    def __init__(self, gen_fn, args):
        self.gen_fn = gen_fn
        self.args = args
        self.score = torch.tensor(0.0)
        self.retval = None
        self.subtraces_trie = MutableChoiceTrie() # values are all `Trace`s

    def _record_subtrace(self, subtrace, address):
        assert isinstance(subtrace, Trace)
        assert isinstance(address, ChoiceAddress)
        if not address:
            # we are recording a subtrace at the empty address
            if self.subtraces_trie.has_choice():
                # there is already a subtrace at the empty address
                raise RuntimeError('The empty address may be visited at most once')
            elif self.subtraces_trie:
                # some other address was also visited
                (other_address, _) = next(self.subtraces_trie.subtries())
                raise RuntimeError('The empty address must be the only address visited, but address '
                                   f'{other_address} was also visited')
            else:
                self.subtraces_trie.set_choice(subtrace)
        else:
            # we are recording a subtrace at a non-empty address
            if self.subtraces_trie.has_choice():
                raise RuntimeError('The empty address must be the only address visited, but address '
                                   f'{address} was also visited')
            if self.subtraces_trie.get_subtrie(address, strict=False):
                # there is a trace at full_address or under full_address
                raise RuntimeError(
                    f'Address {address} was already visited; '
                    'cannot sample a choice at it')
            self.subtraces_trie[address] = subtrace
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
        assert isinstance(subtraces_trie, MutableChoiceTrie)
        assert not subtraces_trie.has_choice()
        # subtraces_trie is a trie where leaves are Traces
        # we need to recursively call get_choice_trie() on those traces
        # and return the expanded choice trie
        trie = MutableChoiceTrie()
        for (address, subtrie) in subtraces_trie.subtries():
            if subtrie.has_choice():
                trace = subtrie.get_choice()
                trie.set_subtrie(address, trace.get_choice_trie())
            else:
                trie.set_subtrie(address, DMLTrace._to_choice_trie(subtrie))
        return trie

    def get_choice_trie(self):
        if self.subtraces_trie.has_choice():
            return self.subtraces_trie.get_choice().get_choice_trie()
        else:
            return DMLTrace._to_choice_trie(self.subtraces_trie)

    @staticmethod
    def _process_deleted_subtraces(discard, prev_subtraces_trie, new_subtraces_trie):
        assert isinstance(prev_subtraces_trie, MutableChoiceTrie)
        assert isinstance(new_subtraces_trie, MutableChoiceTrie)
        if prev_subtraces_trie.has_choice() and not new_subtraces_trie.has_choice():
            # there was a subtrace at the empty address and now there is not
            # there cannot be another subtraces
            assert not prev_subtraces_trie.subtries()
            prev_subtrace = prev_subtraces_trie.get_choice()
            assert not discard.has_choice()
            assert not discard.subtries()
            discard.set_choice(prev_subtrace)
            return prev_subtrace.get_score()
        score = torch.tensor(0.0, requires_grad=False)
        for (address, prev_subtrie) in prev_subtraces_trie.subtries():
            if prev_subtrie.has_choice():
                if not new_subtraces_trie.has_subtrie(address):
                    prev_subtrace = prev_subtrie.get_choice()
                    assert not discard.has_subtrie(address)
                    discard.set_subtrie(address, prev_subtrace.get_choice_trie())
                    score += prev_subtrace.get_score()
            else:
                new_subtrie = new_subtraces_trie.get_subtrie(address, strict=False)
                sub_discard = discard.get_subtrie(address, strict=False)
                score += DMLTrace._process_deleted_subtraces(sub_discard, prev_subtrie, new_subtrie)
                discard.set_subtrie(address, sub_discard)
        return score

    def _get_subtrace_or_none(self, address):
        assert isinstance(address, ChoiceAddress)
        subtrie = self.subtraces_trie.get_subtrie(address, strict=False)
        if subtrie.has_choice():
            return subtrie.get_choice()
        return None

    def update(self, args, args_change, constraints):
        new_trace = DMLTrace(self.get_gen_fn(), args)
        discard = MutableChoiceTrie()
        log_weight = torch.tensor(0.0, requires_grad=False)

        def gentrace(callee, callee_args, address=None):
            assert (address is None) or isinstance(address, ChoiceAddress)
            if isinstance(callee, GenFn):
                if address is None:
                    return _splice_dml_call(callee, callee_args, gentrace)
                nonlocal log_weight
                prev_subtrace = self._get_subtrace_or_none(address)
                if prev_subtrace:
                    prev_callee = prev_subtrace.get_gen_fn()
                    if prev_callee != callee:
                        raise RuntimeError(f'Generative function at address '
                            f'{address} changed from {prev_callee} to {callee}')
                    callee_args_change = tuple(Unknown() for _ in callee_args)
                    (subtrace, log_weight_increment, callee_ret_change, sub_discard) = prev_subtrace.update(
                        callee_args, callee_args_change, constraints.get_subtrie(address, strict=False))
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
            raise RuntimeError(f'Unknown type of generative function: {callee}')

        p = _inject_variables({'gentrace': gentrace}, self.get_gen_fn().p)
        with torch.inference_mode(mode=True):
            new_trace.retval = p(*args)

        log_weight -= DMLTrace._process_deleted_subtraces(
            discard, self.subtraces_trie, new_trace.subtraces_trie)

        return (new_trace, log_weight, Unknown(), discard)

    def regenerate(self, args=None, args_change=None, selection=None):
        raise NotImplementedError()

    def choice_gradients(self, selection, retval_grad):
        if selection is not None:
            raise NotImplementedError()  # TODO add support for gradients wrt choices.
        with torch.inference_mode(mode=False):
            score = torch.tensor(0.0, requires_grad=False)

            def gentrace(callee, callee_args, address=None):
                if isinstance(callee, GenFn):
                    if address is None:
                        return _splice_dml_call(callee, callee_args, gentrace)
                    with torch.inference_mode(mode=True):
                        subtrace = self.subtraces_trie[address]
                        torch_autograd_function = torch_autograd_function_from_trace(subtrace)
                    callee_args_unrolled = gradients.unroll_torch_tensors(callee_args)
                    (score_increment, *callee_retval_unrolled) = torch_autograd_function(*callee_args_unrolled)
                    callee_retval = gradients.roll_torch_tensors(
                            subtrace.get_retval(), tuple(callee_retval_unrolled))
                    nonlocal score
                    score += score_increment
                    return callee_retval
                if isinstance(callee, torch.nn.Module):
                    for param in callee.parameters():
                        param.requires_grad_(False)
                    return callee(*callee_args)
                raise RuntimeError(f'Unknown type of generative function: {callee}')

            p = _inject_variables({'gentrace': gentrace}, self.gen_fn.p)

            args = self.get_args()
            args_tracked = gradients.track(args)
            retval = p(*args_tracked)

            inputs = gradients.unroll_torch_tensors(args_tracked)
            if score.requires_grad:
                outputs = (score, *gradients.unroll_torch_tensors(retval))
                grad_outputs = (torch.tensor(1.0), *gradients.unroll_torch_tensors(retval_grad))
            else:
                outputs = gradients.unroll_torch_tensors(retval)
                grad_outputs = gradients.unroll_torch_tensors(retval_grad)
            input_grads = torch.autograd.grad(
                outputs, inputs,
                grad_outputs=grad_outputs,
                allow_unused=True)
            arg_grads = gradients.roll_torch_tensors(args, input_grads)

        return arg_grads, None, None

    def accumulate_param_gradients(self, retgrad, scale_factor):
        raise NotImplementedError()


def gendml(p):
    return DMLGenFn(p)
