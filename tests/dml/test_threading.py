from pygen.dml.lang import gendml
from pygen.dists import bernoulli
from pygen.choice_address import addr
from pygen.choice_trie import ChoiceTrie, MutableChoiceTrie
from pygen.dml.lang import inline 
from pygen.gfi import get_gentrace
import torch

import threading

@gendml
def f():
    assert get_gentrace()
    return None

@gendml
def g():
    return f() @ inline


def test_threadsafe():

    def do_simulate():
        for j in range(1000):
            assert not get_gentrace()
            g.simulate(())
            assert not get_gentrace()

    n = 10
    runnables = []
    for i in range(n):
        t = threading.Thread(target=do_simulate)
        t.start()
        runnables.append(t)
    
    for i in range(n):
        runnables[i].join()

test_threadsafe()
