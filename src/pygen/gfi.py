import pygen
import torch

class Call:
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args
    def __matmul__(self, address):
        return pygen.thread_local_storage.gentrace(
                self.callee, self.args, address=address)

def set_gentrace(gentrace):
    try:
        prev = pygen.thread_local_storage.gentrace 
    except AttributeError:
        prev = None
    pygen.thread_local_storage.gentrace = gentrace
    return prev

def get_gentrace():
    try:
        return pygen.thread_local_storage.gentrace 
    except AttributeError:
        return None


class TorchModule:
    def __init__(self, obj):
        self._wrapped_obj = obj
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_obj, attr)
    def __call__(self, *args):
        return Call(self._wrapped_obj, args)
    

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
        return Call(self, args)


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
