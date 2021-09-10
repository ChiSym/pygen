from ..gfi import GenFn, Trace
from ..choice_address import ChoiceAddress, addr
from ..choice_trie import ChoiceTrie, MutableChoiceTrie
from functools import wraps
import torch


def _get_subtrie_or_empty(choice_trie, address):
    assert isinstance(choice_trie, ChoiceTrie)
    assert isinstance(address, ChoiceAddress)
    try:
        return choice_trie.get_subtrie(address)
    except RuntimeError:
        return MutableChoiceTrie()


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
    if not isinstance(callee, DMLGenFn):
        raise RuntimeError("Address required when"
                           f" calling a non-DML generative function: {callee}")
    p = _inject_variables({"gentrace": gentrace}, callee.p)
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
                raise RuntimeError("Unknown type of generative function:"
                                   f" {callee}")

        p = _inject_variables({"gentrace": gentrace}, self.p)
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
                else:
                    sub_constraints = _get_subtrie_or_empty(constraints, address)
                    (subtrace, log_weight_increment) = callee.generate(callee_args, sub_constraints)
                    nonlocal log_weight
                    log_weight += log_weight_increment
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
        return (trace, log_weight)

    def __call__(self, *args):
        trace = self.simulate(args)
        return trace.get_retval()


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
            trie[addr()] = self.empty_address_subtrace.get_retval()
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
                    discard.set_subtrie(addr(k), v.get_choice_trie())
            else:
                assert isinstance(v, dict)
                if k in new_subtraces_trie:
                    sub_discard = _get_subtrie_or_empty(discard, k)
                    DMLTrace._add_unvisited_to_discard(sub_discard, v, new_subtraces_trie[v])
                else:
                    d = MutableChoiceTrie()
                    DMLTrace._add_unvisited_to_discard(d, v, new_subtraces_trie[v])
                    discard.set_subtrie(addr(k), d)

    def _get_subtrace(self, subtraces_trie, address):
        assert isinstance(address, ChoiceAddress)
        if address:
            # non-empty address
            first = address.first()
            rest = address.rest()
            if first in subtraces_trie:
                if rest:
                    trie = subtraces_trie[first]
                    if not isinstance(trie, dict):
                        # there was previously a subtrace at this address, but now there is a subtrace at a
                        # an extension of this address
                        return None
                    return self._get_subtrace(subtraces_trie[first], rest)
                else:
                    subtrace = subtraces_trie[first]
                    if not isinstance(subtrace, Trace):
                        # there was previously a subtrace at an extension of this address, but now there this a
                        # subtrace at this address
                        return None
                    return subtrace
            else:
                return None
        else:
            # empty address
            return self.empty_address_subtrace

    def update(self, args, constraints):
        new_trace = DMLTrace(self.get_gen_fn(), args)
        discard = MutableChoiceTrie()
        log_weight = torch.tensor(0.0, requires_grad=False)

        def gentrace(callee, callee_args, address=None):
            assert (address is None) or isinstance(address, ChoiceAddress)
            if isinstance(callee, GenFn):
                if address is None:
                    return _splice_dml_call(callee, callee_args, gentrace)
                else:
                    nonlocal log_weight
                    prev_subtrace = self._get_subtrace(self.subtraces_trie, address)
                    if prev_subtrace:
                        if prev_subtrace.get_gen_fn() != callee:
                            raise RuntimeError(f'Generative function at address {address}'
                                               'changed from {prev_callee} to {callee}')
                        (subtrace, log_weight_increment, sub_discard) = prev_subtrace.update(
                            callee_args, _get_subtrie_or_empty(constraints, address))
                        if sub_discard:
                            discard.set_subtrie(address, sub_discard)
                    else:
                        (subtrace, log_weight_increment) = callee.generate(
                            callee_args, _get_subtrie_or_empty(constraints, address))
                    log_weight += log_weight_increment
                    new_trace._record_subtrace(subtrace, address)
                    return subtrace.get_retval()
            elif isinstance(callee, torch.nn.Module):
                self.get_gen_fn()._record_torch_nn_module(callee)
                return callee(*callee_args)
            else:
                raise RuntimeError('Unknown type of generative function:'
                                   f' {callee}')

        p = _inject_variables({"gentrace": gentrace}, self.get_gen_fn().p)
        with torch.inference_mode(mode=True):
            new_trace.retval = p(*args)

        log_weight -= DMLTrace._deleted_subtraces_score(self.subtraces_trie, new_trace.subtraces_trie)
        self._add_unvisited_to_discard(discard, self.subtraces_trie, new_trace.subtraces_trie)

        return (new_trace, log_weight, discard)

    def regenerate(self, args, constraints):
        raise NotImplementedError()

    def choice_gradients(self, selection, retval_grad):
        raise NotImplementedError()

    def accumulate_param_gradients(self, retgrad, scale_factor):
        raise NotImplementedError()


def gendml(p):
    return DMLGenFn(p)