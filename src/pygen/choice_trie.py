class ChoiceTrie:

    def __init__(self):
        raise NotImplementedError()

    def is_primitive(self, addr=None):
        raise NotImplementedError()

    def get_primitives(self):
        # TODO dict.items() iterator, like get_values_shallow() in Gen.jl
        raise NotImplementedError()

    def get_subtries(self):
        # TODO dict.items() iterator, like get_submaps_shallow() in Gen.jl
        raise NotImplementedError()

    def flatten(self):
        # TODO returns a Python dictionary that has the same interface as a
        # dictionary view, where keys are multi-part addresses
        raise NotImplementedError()

    def __getitem__(self, addr):
        # TODO indexing into a choice map means either retreiving the subtrie
        # (if the sub-choice-trie is not primtive) or the value (if it is
        # primitive)
        raise NotImplementedError()

    def __iter__(self):
        # TODO returns an iterator over (k, v)
        raise NotImplementedError()

# Gen.choicemap((addr1, val1), (addr2, val2), ..)


class MutableChoiceTrie(ChoiceTrie):

    def __init__(self):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        # NOTE: key is an 'Address'
        raise NotImplementedError()

# tests


trie = MutableChoiceTrie({"a": 1, ("b", "c") : 2})
assert not trie.is_primitive()
assert trie.is_primitive("a")
assert trie["a"] == 1
assert trie["b", "c"] == 2
assert trie[("b", "c")] == 2
b_trie = trie["b"]
assert isinstance(b_trie, ChoiceTrie)
assert not b_trie.is_primitive()
assert b_trie.is_primitive("c")
assert b_trie["c"] == 2

# trie = {"b" : {"a" : {(): 1.123}}}
trie = MutableChoiceTrie({("b", "a"), 1.123})
assert not trie.is_primitive("b")
b_trie = trie["b"]
assert b_trie == MutableChoiceTrie({"a": 1.123})

#trie = {"b" : {(): {"a" : {(): 1.123}}}}
trie = MutableChoiceTrie({"b" : MutableChoiceTrie({"a": 1.123})})
assert trie.is_primitive("b")
value = trie["b"]
assert value == MutableChoiceTrie({"a": 1.123})

# TODO implement a copy constructor using this pattern, and test it
def my_walker(trie):
    for (k, sub_trie_or_value) in trie:
        if trie.is_primitive(k):
            value = sub_trie_or_value
            # TODO do something with the value
        else:
            sub_trie = sub_trie_or_value
            # TODO do something with the sub_trie
