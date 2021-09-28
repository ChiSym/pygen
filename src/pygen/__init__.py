import threading
thread_local_storage = threading.local()

def gentrace(*args):
    raise NotImplementedError(("gentrace", args))
