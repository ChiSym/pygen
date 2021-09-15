from pyrsistent import PVector, pvector
from ..choice_address import addr
from ..choice_trie import MutableChoiceTrie, ChoiceTrie
from ..deterministic import CustomChangeGenFn
from ..gfi import GenFn, Trace, Same, Unknown


class ListChange:

    def __init__(self, new_len, prev_len, substitutions):
        self.new_len = new_len
        self.prev_len = prev_len
        self.substitutions = substitutions


# class NewList(CustomChangeGenFn):
#
#     def apply(self, args):
#         return pvector()
#
#     def change(self, prev_args, new_args, args_change, prev_retval):
#         return prev_retval, Same()


class ListTrace(Trace):

    def __init__(self, gen_fn, args, subtraces, retval, score):
        assert isinstance(subtraces, PVector)
        assert isinstance(retval, PVector)
        self.gen_fn = gen_fn
        self.args = args
        self.subtraces = subtraces
        self.retval = retval
        self.score = score
        self.n = len(subtraces)

    def get_gen_fn(self):
        return self.gen_fn

    def get_args(self):
        return self.args

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_choice_trie(self):
        choice_trie = MutableChoiceTrie()
        for (i, subtrace) in enumerate(self.subtraces):
            choice_trie.set_subtrie(addr(i), subtrace.get_choice_trie())
        return choice_trie



    def get_retained_and_constrained(self, new_length, constraints):
        assert isinstance(constraints, ChoiceTrie)
        keys = set()
        for key, subtrie in constraints.subtries():
            assert isinstance(key, int)
            if 0 <= key < new_length:
                keys.add(key)
            else:
                raise RuntimeError(f'invalid address: {key}')
        return keys





class ListUnfoldN(GenFn):

    def __init__(self, kernel):
        self.kernel = kernel

    def simulate(self, args):
        (n, state, *params) = args
        score = 0.0
        retval = pvector()
        subtraces = pvector()
        for i in range(n):
            subtrace = self.kernel.simulate((state, *params))
            state = subtrace.get_retval()
            retval = retval.append(state)
            subtraces = subtraces.append(subtrace)
            score += subtrace.get_score()
        return ListUnfoldNTrace(self, args, subtraces, retval)

    def generate(self, args, constraints):
        # TODO decide on policy for error-checking of constraints
        (n, state, *params) = args
        log_weight = 0.0
        retval = pvector()
        subtraces = pvector()
        for i in range(n):
            callee_args = (state, *params)
            callee_constraints = constraints.get_subtrie(i)
            subtrace, log_weight_increment = self.kernel.generate(callee_args, callee_constraints)
            state = subtrace.get_retval()
            retval = retval.append(state)
            subtraces = subtraces.append(subtrace)
            log_weight += log_weight_increment
        trace = ListUnfoldNTrace(self, args, subtraces, retval)
        return trace, log_weight


from ..change import unpack_tuple_change


# PROBLEMS:
# - list change hint seems weak, a little arbitrary
# - change hints are complicated..
# - the GenCollections code is long

class ListUnfoldNTrace(ListTrace):

    @staticmethod
    def _update_deleted(prev_n, n, subtraces, retval, discard):
        assert n < prev_n
        assert len(subtraces) == prev_n
        assert len(retval) == prev_n
        score_increment = 0.0
        for i in range(n, prev_n):
            subtrace = subtraces[i]
            score_increment -= subtrace.get_score()
            discard.set_subtrie(i, subtrace.get_choice_trie())
        subtraces = subtraces.delete(n, stop=prev_n)
        assert len(subtraces) == n
        retval = retval.delete(n, stop=prev_n)
        return subtraces, retval, score_increment

    @staticmethod
    def _update_new(prev_n, n, args, subtraces, retval):
        pass  # TODO

    @staticmethod
    def _get_to_visit(prev_n, n, params_changed, constraints):
        if params_changed:
            return range(0, min(prev_n, n))
        else:
            constrained_idx = set()
            for i, _ in constraints.subtries():
                constrained_idx.add(i)
            return sorted(constrained_idx)

    def update(self, args, args_change, constraints):
        (n, state, *params) = args
        (n_change, state_change, *param_changes) = unpack_tuple_change(args, args_change)
        subtraces = self.subtraces
        retval = self.retval
        score = self.get_score()
        log_weight = 0.0
        discard = MutableChoiceTrie()

        # handle any removed subtraces
        if n < self.n:
            subtraces, retval, score_increment = ListUnfoldNTrace._update_deleted(
                self.n, n, subtraces, retval, discard)
            score += score_increment
            log_weight += score_increment

        # handle modified subtraces
        to_visit = ListUnfoldNTrace._get_to_visit(n, self.n, params_changed, constraints)

        ListUnfoldNTrace._update_modified()

        # handle any new subtraces
        if n > self.n:
            subtraces, retval, score_increment = ListUnfoldNTrace._update_new(
                self.n, n, args, subtraces, retval)
            score += score_increment
            log_weight += score_increment

        new_trace = ListUnfoldNTrace(self.gen_fn, args, subtraces, retval, score)

        # TODO
        retval_change = ListTrace.compute_retval_change(n, self.n, updated_retval_changes)

        return new_trace, log_weight, retval_change, discard

    def regenerate(self, args, args_change, selection):
        raise NotImplementedError()

    def accumulate_param_gradients(self, retgrad, scale_factor):
        raise NotImplementedError()

    def choice_gradients(self, selection, retgrad):
        raise NotImplementedError()


