class ChoiceTrie:

    def __init__(self):
        raise NotImplementedError()

    def is_primitive(self, key=None):
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
        # TODO indexing into a choice map means either retrieving the subtrie
        # (if the subtrie is not primitive) or the value (if it is primitive)
        raise NotImplementedError()

    def __iter__(self):
        # TODO returns an iterator over (k, v)
        raise NotImplementedError()

# Gen.choicemap((addr1, val1), (addr2, val2), ..)

from pygen.choice_address import ChoiceAddress
from pygen.choice_address import addr

class MutableChoiceTrie(ChoiceTrie):
    def __init__(self, spec=None):
        self.trie = {}
        if spec:
            for k, v in spec.items():
                self[k] = v

    def is_primitive(self, key=None):
        if key is None:
            return () in self.trie
        if key not in self.trie:
            raise RuntimeError('No such key: %s' % (key,))
        return self.trie[key].is_primitive()

    def __getitem__(self, address):
        # Primitive trie.
        if self.is_primitive():
            if address:
                raise IndexError('No such address: %s' % (address,))
            return self.trie[()]
        # Compound trie with primitive child.
        key = address.first()
        if self.is_primitive(key):
            return self.trie[key].trie[()]
        rest = address.rest()
        if not rest:
            return self.trie[key]
        return self.trie[key][rest]

    def __setitem__(self, address, value):
        # NOTE: address is an 'ChoiceAddress'
        assert isinstance(address, ChoiceAddress)
        if self.is_primitive():
            raise RuntimeError('Cannot add choices to a primitive trie.')
        if not address:
            if self.trie:
                raise RuntimeError('Cannot add primitive choice to nonempty trie.')
            self.trie[()] = value
        else:
            key = address.first()
            # Create a subtrie.
            if key not in self.trie:
                self.trie[key] = MutableChoiceTrie()
            # Allow mutating a primitive child.
            if self.trie[key].is_primitive():
                self.trie[key] = MutableChoiceTrie()
            # Set the subtrie.
            rest = address.rest()
            self.trie[key][rest] = value

    def asdict(self):
        if self.is_primitive():
            return dict(self.trie)
        return {k: v.asdict() for k, v in self.trie.items()}

    def __iter__(self):
        return iter(self.trie)

    def __bool__(self):
        return bool(self.trie)

    def __str__(self):
        return str(self.asdict())

    def __eq__(self, x):
        return self.trie == x.trie

import pytest

# Primitive trie.
trie = MutableChoiceTrie()
trie[addr()] = 1
assert trie.is_primitive()
with pytest.raises(RuntimeError):
    trie[addr("a")] = 1
assert trie[addr()] == 1

# Trie with single address.
trie = MutableChoiceTrie()
trie[addr("a")] = 1
assert trie[addr("a")] == 1
assert not trie.is_primitive()
assert trie.is_primitive("a")
with pytest.raises(IndexError):
    trie[addr()]

print(trie)
with pytest.raises(RuntimeError):
    trie[addr()] = 1

trie = MutableChoiceTrie()
trie[addr("a")] = 2
print(trie)
trie[addr("b", "c")] = 3
print(trie)
trie[addr("a", "c")] = 5
print(trie)

print(trie[addr("a")])

t1 = MutableChoiceTrie({addr("a") : {(): 1}})
t2 = MutableChoiceTrie()
t2[addr("a")] = 1
import ipdb; ipdb.set_trace()
assert t1 == t2

# tests
# trie = MutableChoiceTrie({"a": 1, ("b", "c") : 2})
# trie = MutableChoiceTrie({("a",): 1, ("b", "c") : 2})
# trie = MutableChoiceTrie({(("a",),): 1, ("b", "c") : 2})
# trie = MutableChoiceTrie({addr("a"): 1, addr("b", "c") : 2})


# assert not trie.is_primitive()
# assert trie.is_primitive("a")
# assert trie["a"] == 1 # => trie[("a",)]
# assert trie["b", "c"] == 2
# assert trie[("b", "c")] == 2
# b_trie = trie["b"]
# assert isinstance(b_trie, ChoiceTrie)
# assert not b_trie.is_primitive()
# assert b_trie.is_primitive("c")
# assert b_trie["c"] == 2

# trie = {"b" : {"a" : {(): 1.123}}}
# trie = MutableChoiceTrie({("b", "a"), 1.123})
# assert not trie.is_primitive("b")
# b_trie = trie["b"]
# assert b_trie == MutableChoiceTrie({"a": 1.123})

# trie = {"b" : {(): {"a" : {(): 1.123}}}}
# assert trie.is_primitive("b")
# trie = MutableChoiceTrie({"b" : MutableChoiceTrie({"a": 1.123})})
# value = trie["b"]
# assert value == MutableChoiceTrie({"a": 1.123})

# # TODO implement a copy constructor using this pattern, and test it
# def my_walker(trie):
#     for (k, sub_trie_or_value) in trie:
#         if trie.is_primitive(k):
#             value = sub_trie_or_value
#             # TODO do something with the value
#         else:
#             sub_trie = sub_trie_or_value
#             # TODO do something with the sub_trie
