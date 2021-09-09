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

def addressify(x):
    if isinstance(x, tuple):
        return addr(*x)
    return addr(x)

class MutableChoiceTrie(ChoiceTrie):
    def __init__(self, spec=None):
        self.trie = {}
        if spec:
            for k, v in spec.items():
                address = addressify(k)
                self[address] = v

    def is_primitive(self, key=None):
        if key is None:
            return () in self.trie
        if key not in self.trie:
            raise RuntimeError('No such key: %s' % (key,))
        return self.trie[key].is_primitive()

    # def flatten(self):
        # if self.is_primitive():
        #     return {addr(): self.trie[()]}
        # d = {}
        # for k, v in self:
        #     if self.is_primitive(k):
        #         d.update({addr(k): self[k]})
        #     else:
        #         d_sub = flatten(v)
        #         d_sub_prefix = {}
        #     d.update(d)
        # return d
        # pass

    def asdict(self):
        if self.is_primitive():
            return dict(self.trie)
        return {k: v.asdict() for k, v in self.trie.items()}

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

    def __iter__(self):
        return iter(self.trie.items())

    def __bool__(self):
        return bool(self.trie)

    def __str__(self):
        return str(self.asdict())

    def __repr__(self):
        return str(self)

    def __eq__(self, x):
        return isinstance(x, type(self)) and self.trie == x.trie

import pytest

# Tests for addressify
assert addressify(('1', '2')) == addr('1', '2')
assert addressify('1') == addr('1')
assert addressify(('1',)) == addr('1')
assert addressify(('1', ())) == addr('1', ())
assert addressify(()) == addr()

# Primitive trie.
trie = MutableChoiceTrie()
trie[addr()] = 1
assert trie.is_primitive()
with pytest.raises(RuntimeError):
    trie[addr('a')] = 1
assert trie[addr()] == 1
assert trie.asdict() == {(): 1}

# Trie with single address.
trie = MutableChoiceTrie()
trie[addr('a')] = 1
assert trie[addr('a')] == 1
assert not trie.is_primitive()
assert trie.is_primitive('a')
with pytest.raises(IndexError):
    trie[addr()]
assert trie.asdict() == {'a': {(): 1}}

# Trie with tuples as the keys.
for k in [('a',), (('a', 'b'),)]:
    trie = MutableChoiceTrie()
    trie[addr(k)] = 10
    assert trie.asdict() == {k: {(): 10}}
    assert not trie.is_primitive()
    assert trie.is_primitive(k)

trie = MutableChoiceTrie()
# Create a primitive.
trie[addr('a')] = 2
assert trie.asdict() == {'a': {(): 2}}
# Create a compound.
trie[addr('b', 'c')] = 3
assert trie.asdict() == {'a': {(): 2}, 'b': {'c': {(): 3}}}
# Extend a compound.
trie[addr('b', 'd')] = 14
assert trie.asdict() == {'a': {(): 2}, 'b': {'c': {(): 3}, 'd': {(): 14}}}
# Overwrite a primitive.
trie[addr('a', 'c')] = 5
# Confirm values.
assert trie[addr('a', 'c')] == 5
assert trie[addr('b', 'c')] == 3
assert trie[addr('b', 'd')] == 14
# Overwrite a compound.
trie[addr('b', 'c')] = 13
assert trie[addr('b', 'c')] == 13

# Check initializations agree.
t1 = MutableChoiceTrie({'a': 1})
t2 = MutableChoiceTrie()
t2[addr('a')] = 1
assert t1 == t2

# Check singleton tuple has no effect.
t = MutableChoiceTrie({'a': 2})
assert t == MutableChoiceTrie({('a',): 2})
assert t[addr('a')] == 2
assert t.asdict() == {'a': {(): 2}}
assert not t.is_primitive()
assert t.is_primitive('a')
with pytest.raises(IndexError):
    t[addr()]

# Directly create a primitive trie.
t = MutableChoiceTrie({(): 2})
assert t[()] == 2
assert t.is_primitive()
assert t.asdict() == {(): 2}

# Miscellaneous test.
trie = MutableChoiceTrie({'a': 1, ('b', 'c') : 2})
assert not trie.is_primitive()
assert trie.is_primitive('a')
assert trie[addr('a')] == 1
assert trie[addr('b', 'c')] == 2
b_trie = trie[addr('b')]
assert not b_trie.is_primitive()
assert b_trie.is_primitive('c')
assert b_trie[addr('c')] == 2

# Test ChoiceTrie as values
trie = MutableChoiceTrie({('b', 'a'): 1.123})
assert not trie.is_primitive('b')
assert trie[addr('b')] == MutableChoiceTrie({'a': 1.123})
assert trie.asdict() == {'b': {'a': {(): 1.123}}}

trie = MutableChoiceTrie({'b': {'a' : {(): 1.123}}})
assert trie.is_primitive('b')
assert trie[addr('b')] == {'a' : {(): 1.123}}
d1 = trie.asdict()

trie = MutableChoiceTrie({'b': MutableChoiceTrie({'a': 1.123})})
assert trie.is_primitive('b')
assert trie[addr('b')] == MutableChoiceTrie({'a': 1.123})
d2 = trie.asdict()

assert str(d1) == str(d2)
assert d1 != d2


# trie = MutableChoiceTrie({(): 1})
# assert trie.flatten() == {addr(): 1}

# trie = MutableChoiceTrie({'a': 1})
# assert trie.flatten() == {addr('a'): 1}

choices = {'a': 1.123, ('b', 'c'): 10, ('b', 'e', 1): 11}
trie = MutableChoiceTrie(choices)
assert trie.asdict() == {
    'a':
        {(): 1.123},
    'b': {
        'c': {(): 10},
        'e':
            {1: {(): 11}}}
    }
with pytest.raises(Exception):
    assert trie.flatten() == {
        addr('a'): 1.123,
        addr('b', 'c') : 10,
        addr('b', 'e', 1): 11
    }

x = MutableChoiceTrie()
x[addr('a')] = MutableChoiceTrie()
assert(x.is_primitive('a'))

# TEST CASE FROM MARCO
# y = x[addr("a")]
# y[addr("b")] = 1.123
# print(x.asdict())
# print(y.asdict())
# print(x[addr('a', 'b')])
# assert x[addr('a', 'b')] == 1.123

# get_subtries does not unbox primitives
# __getindex__ does unbox primitives

# set_subtries writes an subtrie
# __setindex__ writes a value (possibly a subtrie)

# TODO implement a copy constructor using this pattern, and test it
# def my_walker(trie):
#     for (k, sub_trie_or_value) in trie:
#         if trie.is_primitive(k):
#             value = sub_trie_or_value
#             # TODO do something with the value
#         else:
#             sub_trie = sub_trie_or_value
#             # TODO do something with the sub_trie
