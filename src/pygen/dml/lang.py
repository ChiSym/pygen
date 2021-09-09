from ..gfi import GenFn, Trace
from ..address import ChoiceAddress
from ..choice_trie import ChoiceTrie
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
    if not isinstance(gen_fn, DMLGenFn):
        raise RuntimeError("Address required when"
            f" calling a non-DML generative function: {callee}")
    if addr is not None:
        raise RuntimeError(f"Address must not be provided for a DML call, got: {addr}")
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
            assert (addr is None) or isinstance(addr, ChoiceAddress)
            if isinstance(callee, GenFn):
                if addr is None:
                    return splice_dml_call(callee, args, addr, gentrace)
                else:
                    subtrace = callee.simulate(args)
                    trace._record_subtrace(subtrace, addr)
                    return subtrace.get_retval()
            elif isinstance(callee, torch.nn.Module):
                self._record_torch_nn_module(callee)
                return callee(*args)
            else:
                raise RuntimeError("Unknown type of generative function:"
                    f" {callee}")

        p = _inject_variables({"gentrace" : gentrace}, self.p)
        with torch.inference_mode(mode=True):
            trace.retval = p(*args)
        return trace

    def generate(self, args, constraints):
        assert isinstance(constraints, ChoiceTrie)
        trace = DMLTrace(self, args)
        log_weight = torch.tensor(0.0, requires_grad=False)

        def gentrace(callee, args, addr=None):
            assert (addr is None) or isinstance(addr, ChoiceAddress)
            if isinstance(callee, GenFn):
                if addr is None:
                    return splice_dml_call(callee, args, addr, gentrace)
                else:
                    # constraints = {"a" : { "b" : { () : 1.123 } } }
                    # assert constraints.get_subtrie(("a",)) == { "b" : { () : 1.123 } }
                    # assert constraints.get_subtrie(("a", "b")) == { () : 1.123 }
                    # constraints.get_subtrie(("a", "b", "c")) # ERROR
                    # constraints.get_subtrie(("a", "b", ())) # ERROR
                    (subtrace, log_weight_incr) = callee.generate(args, constraints.get_subtrie(addr))
                    nonlocal log_weight
                    log_weight += log_weight_incr
                    trace._record_subtrace(subtrace, addr)
                    return subtrace.get_retval()
            elif isinstance(callee, torch.nn.Module):
                self._record_torch_nn_module(callee)
                return callee(*args)
            else:
                raise RuntimeError("Unknown type of generative function:"
                    f" {callee}")

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
        self.empty_address_subtrace = None
        self.subtraces_trie = {}

    @staticmethod
    def _record_subtrace_in_subtraces_trie(subtraces_trie, addr, subtrace, full_addr):
        assert isinstance(addr, ChoiceAddress) and addr
        first = addr.first()
        rest = addr.rest()
        if not rest:
            if first in self.subtraces_trie:
                raise RuntimeError(f'Address {full_addr} already visited; cannot sample a choice at it')
            self.subtraces_trie[first] = subtrace
        else:
            if first not in subtraces_trie:
                subtraces_trie[first] = {}
            DMLTrace._record_subtrace_in_subtraces_trie(subtraces_trie[first], rest, subtrace, full_addr)

    def _record_subtrace(self, subtrace, addr):
        assert isinstance(subtrace, Trace)
        assert isinstance(addr, ChoiceAddress)
        value = subtrace.get_retval()
        assert not value.requires_grad
        if not addr:
            if (self.empty_address_subtrace is not None) or subtraces_trie:
                raise RuntimeError('the empty address may be visited at most once,'
                    f' and must be the only address visited, but address {next(iter(subtraces_trie))}'
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
        # subtraces_trie is a trie where leafs are Traces
        assert isinstance(subtraces_trie, dict)
        trie = MutableChoiceTrie()
        for (k, v) in subtraces_trie.items():
            if isinstance(v, Trace):
                trie.set_subtrie(addr(k), v.get_choice_trie())
            else:
                assert isinstance(v, dict)
                trie.set_subtrie(addr(k), DMLTrace._to_choice_trie(v))

    def get_choice_trie(self):
        if self.empty_address_subtrace is not None:
            trie = MutableChoiceTrie()
            trie[addr()] = self.empty_address_subtrace.get_retval()
            return trie
        else:
            DMLTrace._to_choice_trie(self.subtraces_trie)

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
                    score += _deleted_subtraces_score(v, new_subtraces_trie[v])
                else:
                    score += _deleted_subtraces_score(v, {})
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
                    _add_unvisited_to_discard(discard.get_subtrie(k), v, new_subtraces_trie[v])
                else:
                    d = MutableChoiceTrie()
                    _add_unvisited_to_discard(d, v, new_subtraces_trie[v])
                    discard.set_subtrie(addr(k), d)

    def update(self, args, constraints):
        new_trace = DMLTrace(self.get_gen_fn(), args)
        discard = MutableChoiceTrie()
        log_weight = torch.tensor(0.0, requires_grad=False)

        def gentrace(callee, args, addr=None):
            assert (addr is None) or isinstance(addr, ChoiceAddress)
            if isinstance(callee, GenFn):
                if addr is None:
                    return splice_dml_call(callee, args, addr, gentrace)
                else:
                    nonlocal log_weight
                    if self._has_subtrace(addr):
                        prev_callee = self._get_subtrace(addr)
                        if prev_callee != callee:
                            raise RuntimeError(f'Generative function at address {addr}'
                                'changed from {prev_callee} to {callee}')
                        (subtrace, log_weight_incr, sub_discard) = self._get_subtrace(addr).update(
                            args, constraints.get_subtrie(addr))
                        discard.set_subtrie(addr, sub_discard)
                    else:
                        (subtrace, log_weight_incr) = callee.generate(
                            args, constraints.get_subtrie(addr))
                    log_weight += log_weight_incr
                    trace._record_subtrace(subtrace, addr)
                    return subtrace.get_retval()
            elif isinstance(callee, torch.nn.Module):
                self._record_torch_nn_module(callee)
                return callee(*args)
            else:
                raise RuntimeError('Unknown type of generative function:'
                    f' {callee}')

        p = _inject_variables({"gentrace" : gentrace}, self.get_gen_fn().p)
        with torch.inference_mode(mode=True):
            new_trace.retval = p(*args)

        log_weight -= DMLTrace._deleted_subtrace_score(self.subtraces_trie, new_trace.subtraces_trie)
        self._add_unvisited_to_discard(discard, new_trace)

        return (new_trace, log_weight, discard)


    def regenerate(self, args, constraints):
        raise NotImplementedError()

    def choice_gradients(self, selection, retval_grad):
        raise NotImplementedError()

    def accumulate_param_gradients(self, retgrad, scale_factor):
        raise NotImplementedError()

def gendml(p):
    return DMLGenFn(p)
