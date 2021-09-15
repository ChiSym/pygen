# TODO use abc.ABC?

class GenFn:

    def simulate(self, args):
        raise NotImplementedError()

    def generate(self, args, constraints):
        raise NotImplementedError()

    def propose(self, args):
        trace = self.simulate(args)
        choice_trie = trace.get_choice_trie()
        weight = trace.get_score()
        retval = trace.get_retval()
        return (choice_trie, weight, retval)

    def assess(self, args, constraints):
        (trace, weight) = self.generate(args, constraints)
        retval = trace.get_retval()
        return (weight, retval)


class Trace:

    def get_gen_fn(self):
        raise NotImplementedError()

    def get_args(self):
        raise NotImplementedError()

    def get_retval(self):
        raise NotImplementedError()

    def get_score(self):
        raise NotImplementedError()

    def get_choice_trie(self):
        raise NotImplementedError()

    # TODO: make each argument optional (but if args_change is provided then args must also
    def update(self, args, args_change, constraint):
        raise NotImplementedError()

    # TODO: make each argument optional (but if args_change is provided then args must also
    def regenerate(self, args, args_change, selection):
        raise NotImplementedError()

    def accumulate_param_gradients(self, retgrad, scale_factor):
        raise NotImplementedError()

    def choice_gradients(self, selection, retgrad):
        raise NotImplementedError()
