from .gfi import GenFn, Trace
from .choice_trie import MutableChoiceTrie


class CustomChangeCustomGradTrace(Trace):

    def __init__(self, gen_fn, args, retval):
        assert isinstance(gen_fn, CustomChangeGenFn)
        self.gen_fn = gen_fn
        self.args = args
        self.retval = retval

    def get_gen_fn(self):
        raise NotImplementedError()

    def get_args(self):
        return self.args

    def get_retval(self):
        return self.retval

    def get_score(self):
        return 0.0

    def get_choice_trie(self):
        return MutableChoiceTrie()

    def update(self, args, args_change, constraints):
        new_retval, ret_change = self.gen_fn.change(self.args, args, args_change, self.retval)
        new_trace = CustomChangeCustomGradTrace(self.gen_fn, args, new_retval)
        # TODO: there are lots of cases like this we want to return an empty choice trie
        # it would probably be much more performant to use None, than MutableChoiceTrie()
        return (new_trace, 0.0, ret_change, MutableChoiceTrie())

    def regenerate(self, args, args_change, selection):
        new_retval, ret_change = self.gen_fn.change(self.args, args, args_change, self.retval)
        new_trace = CustomChangeCustomGradTrace(self.gen_fn, args, new_retval)
        # TODO: there are lots of cases like this we want to return an empty choice trie
        # it would probably be much more performant to use None, than MutableChoiceTrie()
        return (new_trace, 0.0, ret_change)

    def accumulate_param_gradients(self, retval_grad, scale_factor):
        args_grad = self.gen_fn.grad(self.args, self.retval, retval_grad)
        choice_dict, grad_dict = None, None
        return args_grad, choice_dict, grad_dict

    def choice_gradients(self, selection, retval_grad):
        args_grad = self.gen_fn.grad(self.args, self.retval, retval_grad)
        choice_dict, grad_dict = None, None
        return args_grad, choice_dict, grad_dict


class CustomChangeGenFn(GenFn):

    def apply(self, args):
        raise NotImplementedError()

    def change(self, prev_args, new_args, args_change, prev_retval):
        # return new value and retdiff
        raise NotImplementedError()

    def grad(self, args, retval, retval_grad):
        return tuple(None for _ in args)

    def simulate(self, args):
        retval = self.apply(args)
        return CustomChangeCustomGradTrace(self, args, retval)

    def generate(self, args, constraints):
        retval = self.apply(args)
        trace = CustomChangeCustomGradTrace(self, args, retval)
        return (trace, 0.0)