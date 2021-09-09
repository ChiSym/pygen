class ChoiceTrie:

    def __init__(self):
        raise NotImplementedError()

    def is_primitive(self):
        """Return True if this trie has exactly one choice, stored under `addr()`."""
        raise NotImplementedError()

    def get_subtrie(self, address):
        """Return the trie under the given `address`."""
        raise NotImplementedError()

    def flatten(self):
        """Returns a `(address, choice)` iterator, where `address` is a
        full path down the trie and `choice` is an choice value."""
        raise NotImplementedError()

    def __getitem__(self, address):
        """Indexing into a choice map means retrieving the subtrie stored
         at `address` (if the subtrie is not primitive) or retrieving the
         choice (if the subtrie is primitive, i.e., unboxing)."""
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

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

    def set_subtrie(self, address, subtrie):
        """Replace the entire subtrie at `address` with the given `subtrie`."""
        assert isinstance(subtrie, ChoiceTrie)
        if self.is_primitive():
            raise RuntimeError('Cannot set_subtrie of a primitive MutableChoiceTrie.')
        key = address.first()
        rest = address.rest()
        if not rest:
            self.trie[key] = subtrie
        else:
            if key not in self.trie:
                self.trie[key] = MutableChoiceTrie()
            # Overwrite primitive subtrie with a compound.
            if self.trie[key].is_primitive():
                self.trie[key] = MutableChoiceTrie()
            # Recurse.
            self.trie[key].set_subtrie(rest, subtrie)

    def get_shallow_choices(self):
        for k, subtrie in self.trie.items():
            if subtrie.is_primitive():
                yield (k, subtrie[addr()])

    def get_shallow_subtries(self):
        for k, subtrie in self.trie.items():
            if not subtrie.is_primitive():
                yield (k, subtrie)

    def flatten(self):
        if self.is_primitive():
            return {addr(): self.trie[()]}
        d = {}
        for k, subtrie in self.trie.items():
            if subtrie.is_primitive():
                d.update({addr(k): subtrie[addr()]})
            else:
                subtrie_flat = subtrie.flatten()
                d_sub_prefix = {addr(k) + t: v for t, v in subtrie_flat.items()}
                d.update(d_sub_prefix)
        return d

    def asdict(self):
        if self.is_primitive():
            return dict(self.trie)
        return {k: v.asdict() for k, v in self.trie.items()}

    def __getitem__(self, address):
        assert isinstance(address, ChoiceAddress)
        # Primitive trie.
        if self.is_primitive():
            if address:
                raise IndexError('No such address: %s' % (address,))
            return self.trie[()]
        subtrie = self.get_subtrie(address)
        # Primitive subtrie: unbox
        if subtrie.is_primitive():
            return subtrie[addr()]
        # Compound subtrie.
        return subtrie

    def __setitem__(self, address, value):
        assert isinstance(address, ChoiceAddress)
        # Write to primitive trie.
        if self.is_primitive():
            # Cannot add new choices.
            if address:
                raise RuntimeError('Cannot add choices to a primitive trie.')
            # Overwrite the choice.
            self.trie[()] = value
            return
        # Write to compound trie.
        elif not address:
            if self.trie:
                raise RuntimeError('Cannot add primitive choice to nonempty trie.')
            self.trie[()] = value
        else:
            key = address.first()
            # Create a subtrie.
            if key not in self.trie:
                self.trie[key] = MutableChoiceTrie()
            # Overwrite primitive subtrie.
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
        # TODO: Return a representation that makes it clear this object
        # is a MutableChoiceTrie, not a dictionary! Once a constructor
        # is designed it will be easier to standardize the representation.
        # return 'MutableChoiceTrie(%s)' % (repr(self.trie),)
        return str(self)

    def __eq__(self, x):
        if isinstance(x, type(self)):
            return self.trie == x.trie
        return False

    @staticmethod
    def copy(x):
        if x.is_primitive():
            trie = MutableChoiceTrie()
            trie[addr()] = x[addr()]
            return trie

        trie = MutableChoiceTrie()
        for k, subtrie in x:
            if subtrie.is_primitive():
                value = subtrie[addr()]
                trie[addr(k)] = value # deepcopy the value?
            else:
                subtrie_recursive = MutableChoiceTrie.copy(subtrie)
                trie.set_subtrie(addr(k), subtrie_recursive)
        return trie

import pytest

# Empty trie.
trie = MutableChoiceTrie()
assert not trie.is_primitive()
assert trie.flatten() == {}
assert trie.asdict() == {}
assert MutableChoiceTrie.copy(trie) == trie

# Primitive trie.
trie = MutableChoiceTrie()
trie[addr()] = 1
assert trie.is_primitive()
assert trie.asdict() == {(): 1}
assert MutableChoiceTrie.copy(trie) == trie

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
assert subtrie[addr()] == 3
# Extend a compound.
trie[addr('b', 'd')] = 14
assert trie.asdict() == {'a': {(): 2}, 'b': {'c': {(): 3}, 'd': {(): 14}}}
assert trie[addr('b')] == trie.get_subtrie(addr('b'))
assert trie[addr('b')].asdict() == {'c': {(): 3}, 'd': {(): 14}}
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

# Writing a dict versus ChoiceTrie as primitive choice.
# = Write a dict.
trie_value = {'a' : {(): 1.123}}
trie = MutableChoiceTrie()
trie[addr('b')] = trie_value
assert trie.get_subtrie(addr('b')).is_primitive()
assert trie[addr('b')] == trie_value
d1 = trie.asdict()
assert d1 == {'b': {(): trie_value}}
# = Write a MutableChoiceTrie.
trie_value = MutableChoiceTrie()
trie_value[addr('a')] = 1.123
trie = MutableChoiceTrie()
trie[addr('b')] = trie_value
assert trie.get_subtrie(addr('b')).is_primitive()
assert trie[addr('b')] == trie_value
assert trie.get_subtrie(addr('b'))[addr()] == trie_value
d2 = trie.asdict()
assert d2 == {'b': {(): trie_value}}
# = Confirm they are not equal
assert d1 != d2

trie = MutableChoiceTrie()
assert trie.flatten() == {}
trie[addr()] = 1
assert trie.flatten() == {addr(): 1}

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
assert trie.flatten() == {
    addr('a'): 1.123,
    addr('b', 'c') : 10,
    addr('b', 'e', 1): 11
}
assert MutableChoiceTrie.copy(trie) == trie

# Setting a subtrie versus setting a primitive using index.
# = setting a primitive that happens to be a choice trie
trie = MutableChoiceTrie()
trie[addr('a')] = MutableChoiceTrie()
assert trie.asdict() == {'a': {(): MutableChoiceTrie()}}
assert trie.flatten() == {addr('a'): MutableChoiceTrie()}
# = setting a subtrie proper
trie.set_subtrie(addr('a'), MutableChoiceTrie())
assert trie.trie == {'a': MutableChoiceTrie()}
assert trie.asdict() == {'a': {}}
assert trie.flatten() == {}

# Flatten a non-empty subtrie with no choices.
trie = MutableChoiceTrie()
trie.set_subtrie(addr('a'), MutableChoiceTrie())
assert trie.flatten() == {}

# Fail to set subtrie of a primitive
trie = MutableChoiceTrie()
trie[addr()] = 1
with pytest.raises(RuntimeError):
    trie.set_subtrie(addr(), MutableChoiceTrie())

trie = MutableChoiceTrie()
trie[addr('a')] = 1
subtrie = MutableChoiceTrie()
subtrie[addr('c')] = 'foo'
trie.set_subtrie(addr('a'), subtrie)
assert trie.get_subtrie(addr('a')) == subtrie
trie.set_subtrie(addr('a', 'b'), subtrie)
assert trie.get_subtrie(addr('a', 'b')) == subtrie
# XXX Warning: Cannot print(subtrie), infinite recursion.

# XXX Warning: Cannot print(trie), infinite recursion.
trie = MutableChoiceTrie()
trie.set_subtrie(addr('a'), trie)

trie = MutableChoiceTrie()
trie.set_subtrie(addr('a'), MutableChoiceTrie())
subtrie = trie.get_subtrie(addr('a'))
subtrie[addr('b')] = 1.123
assert trie[addr('a', 'b')] == 1.123
