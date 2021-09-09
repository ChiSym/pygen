class ChoiceTrie:

    def __init__(self):
        raise NotImplementedError()

    def is_primitive(self):
        raise NotImplementedError()

    def get_subtrie(self, address):
        # the address is an arbitrary address, and where the result
        # is a ChoiceTrie (primitive or not).
        raise NotImplementedError()

    def set_subtrie(self, key, trie):
        raise NotImplementedError

    def flatten(self):
        # TODO returns a Python dictionary that has the same interface as a
        # dictionary view, where keys are multi-part addresses
        raise NotImplementedError()

    def get_primitives(self):
        # TODO dict.items() iterator, like get_values_shallow() in Gen.jl
        raise NotImplementedError()

    def get_subtries(self):
        # TODO dict.items() iterator, like get_submaps_shallow() in Gen.jl
        raise NotImplementedError()

    def __getitem__(self, addr):
        # TODO indexing into a choice map means either retrieving the subtrie
        # (if the subtrie is not primitive) or the value (if it is primitive)
        raise NotImplementedError()

    def __setitem__(self, addr, value):
        # Records a primitive choice at addr.
        raise NotImplementedError()

    def __iter__(self):
        # TODO returns an iterator over (k, v)
        raise NotImplementedError()

# Gen.choicemap((addr1, val1), (addr2, val2), ..)

from pygen.choice_address import ChoiceAddress
from pygen.choice_address import addr

class MutableChoiceTrie(ChoiceTrie):
    def __init__(self):
        self.trie = {}

    def is_primitive(self):
        if () in self.trie:
            assert len(self.trie) == 1
            return True
        return False

    def get_subtrie(self, address):
        if self.is_primitive():
            raise RuntimeError('Cannot get_subtrie of primitive MutableChoiceTrie.')
        key = address.first()
        if key not in self.trie:
            raise IndexError('No such address: %s' % (address,))
        rest = address.rest()
        if rest:
            return self.trie[key].get_subtrie(rest)
        else:
            return self.trie[key]

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
        # Compound trie
        key = address.first()
        if self.trie[key].is_primitive():
            return self.trie[key].trie[()]
        rest = address.rest()
        if not rest:
            return self.trie[key]
        return self.trie[key][rest]

    def __setitem__(self, address, value):
        assert isinstance(address, ChoiceAddress)
        if self.is_primitive():
            # Cannot add new choices.
            if address:
                raise RuntimeError('Cannot add choices to a primitive trie.')
            # Overwrite the primitive choice.
            self.trie[()] = value
            return
        # Write to a compound trie.
        elif not address:
            if self.trie:
                raise RuntimeError('Cannot add primitive choice to nonempty trie.')
            self.trie[()] = value
        else:
            key = address.first()
            # Create a subtrie.
            if key not in self.trie:
                self.trie[key] = MutableChoiceTrie()
            # Overwrite primitive child with a compound.
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

# Primitive trie.
trie = MutableChoiceTrie()
trie[addr()] = 1
assert trie.is_primitive()
assert trie.asdict() == {(): 1}

# Cannot write to a primitive trie.
with pytest.raises(RuntimeError):
    trie[addr('a')] = 1

# Can overwrite primitive trie.
trie[addr()] = 2
assert trie[addr()] == 2
assert trie.asdict() == {(): 2}

# Cannot get subtrie of primitive trie.
with pytest.raises(RuntimeError):
    trie.get_subtrie(addr())

# {'a': {(): 2}}
# {'a': {'b': {(): 2}}}
# {'a': {'b': {(): 1}, {'c' : {(), 1}}}}

# Trie with single address.
trie = MutableChoiceTrie()
trie[addr('a')] = 1
assert trie[addr('a')] == 1
assert not trie.is_primitive()
with pytest.raises(IndexError):
    trie[addr()]
assert trie.asdict() == {'a': {(): 1}}
subtrie = trie.get_subtrie(addr('a'))
assert subtrie.is_primitive()
assert subtrie[addr()] == 1

# Trie with tuples as the keys.
for k in [('a',), (('a', 'b'),)]:
    trie = MutableChoiceTrie()
    trie[addr(k)] = 10
    assert trie.asdict() == {k: {(): 10}}
    assert not trie.is_primitive()
    subtrie = trie.get_subtrie(addr(k))
    assert subtrie.is_primitive()

trie = MutableChoiceTrie()
# Create a primitive.
trie[addr('a')] = 2
assert trie.asdict() == {'a': {(): 2}}
# Create a compound.
trie[addr('b', 'c')] = 3
assert trie.asdict() == {'a': {(): 2}, 'b': {'c': {(): 3}}}
subtrie = trie.get_subtrie(addr('b'))
assert not subtrie.is_primitive()
subtrie = trie.get_subtrie(addr('b', 'c'))
assert subtrie.is_primitive()
# Extend a compound.
trie[addr('b', 'd')] = 14
assert trie.asdict() == {'a': {(): 2}, 'b': {'c': {(): 3}, 'd': {(): 14}}}
with pytest.raises(IndexError):
    trie.get_subtrie(addr('d'))
subtrie = trie.get_subtrie(addr('b', 'd'))
assert subtrie.is_primitive()
assert subtrie[addr()] == 14
# Overwrite a primitive with a compound.
trie[addr('a', 'c')] = 5
subtrie = trie.get_subtrie(addr('a'))
assert not subtrie.is_primitive()
assert subtrie[addr('c')] == 5
# Confirm values.
assert trie[addr('a', 'c')] == 5
assert trie[addr('b', 'c')] == 3
assert trie[addr('b', 'd')] == 14
# Overwrite a compound.
trie[addr('b', 'c')] = 13
assert trie[addr('b', 'c')] == 13

# Write a dict as a value.
trie = MutableChoiceTrie()
trie[addr('b')] = {'a' : {(): 1.123}}
assert trie.get_subtrie(addr('b')).is_primitive()
assert trie[addr('b')] == {'a' : {(): 1.123}}
d1 = trie.asdict()
assert d1 == {'b': {(): {'a' : {(): 1.123}}}}

# Write a MutableChoiceTrie as a value.
trie = MutableChoiceTrie()
trie_value = MutableChoiceTrie()
trie_value[addr('a')] = 1.123
trie[addr('b')] = trie_value
assert trie.get_subtrie(addr('b')).is_primitive()
assert trie[addr('b')] == trie_value
assert trie.get_subtrie(addr('b'))[addr()] == trie_value
d2 = trie.asdict()

assert str(d1) == str(d2)
assert d1 != d2

# trie = MutableChoiceTrie({(): 1})
# assert trie.flatten() == {addr(): 1}

# trie = MutableChoiceTrie({'a': 1})
# assert trie.flatten() == {addr('a'): 1}

trie = MutableChoiceTrie()
trie[addr('a')] = 1.123
trie[addr('b', 'c')] = 10
trie[addr('b', 'e', 1)] = 11
assert trie.get_subtrie(addr('b', 'e'))[addr(1)] == 11

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

# New requirements
# get_subtrie where the address is an arbitrary address, and where the result can be a primitive or non-primitive choice trie
# is_primitive() -- zero-argument version
# trie[addr()] -- getting the value from a primitive choice trie
# set_subtrie(k, v) where k is a single element of an address and where v is a ChoiceTrie
# That last point reminds me -- we need a copy constructor, where we construct a MutableChoiceTrie from a ChoiceTrie

# TODO implement a copy constructor using this pattern, and test it
# def my_walker(trie):
#     for (k, sub_trie_or_value) in trie:
#         if trie.is_primitive(k):
#             value = sub_trie_or_value
#             # TODO do something with the value
#         else:
#             sub_trie = sub_trie_or_value
#             # TODO do something with the sub_trie
