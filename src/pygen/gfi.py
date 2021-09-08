class GenFn:

    def simulate(self, args):
        raise NotImplementedError()

    def generate(self, args, constraints):
        raise NotImplementedError()

    def propose(self, args):
        trace = self.simulate(args)
        choices = trace.get_choices()
        weight = trace.get_score()
        retval = trace.get_retval()
        return (choices, weight, retval)

    def assess(self, args, constraints):
        (trace, weight) = self.generate(args, constraints)
        retval = trace.get_retval()
        return (weight, retval)

class Trace:

    def get_args(self):
        raise NotImplementedError()

    def get_retval(self):
        raise NotImplementedError()

    def get_score(self):
        raise NotImplementedError()

    def get_choices(self):
        raise NotImplementedError()

    def update(self, args, constraints):
        raise NotImplementedError()

    def regenerate(self, args, selection):
        raise NotImplementedError()

    def accum_param_grads(self, retgrad, scale_factor):
        raise NotImplementedError()

    def choice_gradients(self, selection, retgrad):
        raise NotImplementedError()
