# TODO replace with thread-local version using threading.local()
gentrace = None
class Inline:
    pass
inline = None

class AppliedGenFn:
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args
    def __matmul__(self, address):
        global gentrace
        return gentrace(self.callee, self.args, address=address)

def set_gentrace(gentrace_new):
    global gentrace
    gentrace = gentrace_new
    


###########

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

    def __call__(self, *args):
        return AppliedGenFn(self, args)


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

    def update(self, args, constraints):
        raise NotImplementedError()

    def regenerate(self, args, selection):
        raise NotImplementedError()

    def accumulate_param_gradients(self, retgrad, scale_factor):
        raise NotImplementedError()

    def choice_gradients(self, selection, retgrad):
        raise NotImplementedError()
