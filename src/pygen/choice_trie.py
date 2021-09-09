from .choice_address import ChoiceAddress
from .choice_address import addr

class ChoiceTrie:
    """Implements a Trie (prefix tree) data structure.
    https://en.wikipedia.org/wiki/Trie

    The keys in the trie tree are instances of ChoiceAddress.
    The leaves (choices) in the trie are arbitrary Python values.
    """

    def __init__(self):
        raise NotImplementedError()

    def is_primitive(self):
        """Return True if and only if this trie has exactly one choice,
        stored under the empty address `addr()`."""
        raise NotImplementedError()

    def get_subtrie(self, address):
        """Return the trie under the given `address`."""
        raise NotImplementedError()

    def get_choice(self, address):
        """Return the choice under the given `address`."""
        raise NotImplementedError()

    def flatten(self):
        """Returns an iterator over all `(address, choice)` pairs in the trie
        that satisfy `get_choice(address) === choice`."""
        raise NotImplementedError()

    def __getitem__(self, address):
        """Indexing into a ChoiceTrie returns either a ChoiceTrie or a choice.

        1. If the ChoiceTrie is primitive, equivalent to `get_choice(address)`.

        2. If the ChoiceTrie is non primitive, returns `get_choice(address)`
        if there exists a choice under `address` and returns `get_submap(address)`
        otherwise.
        """
        raise NotImplementedError()

    def __iter__(self):
        """Returns an iterator over `(address, v)` pairs that
        satisfy `subtrie[address] = v`."""
        raise NotImplementedError()

class MutableChoiceTrie(ChoiceTrie):
    def __init__(self):
        self.trie = {}

    def is_primitive(self):
        b = () in self.trie
        assert not b or len(self.trie) == 1
        return b

    def get_choice(self, address):
        assert isinstance(address, ChoiceAddress)
        # Primitive trie.
        if self.is_primitive():
            if address:
                raise RuntimeError(f'No choice under address {address}')
            return self.trie[()]
        # Compound trie.
        if not address:
            raise RuntimeError(f'No choice under address: {address}')
        key = address.first()
        if key not in self.trie:
            raise RuntimeError(f'No choice under address: {address}')
        rest = address.rest()
        return self.trie[key].get_choice(rest)

    def get_subtrie(self, address):
        assert isinstance(address, ChoiceAddress)
        # Primitive trie.
        if self.is_primitive():
            raise RuntimeError('Cannot get_subtrie of primitive MutableChoiceTrie.')
        # Compound trie.
        if not address:
            raise RuntimeError('Cannot get_subtrie at empty address.')
        key = address.first()
        if key not in self.trie:
            raise RuntimeError(f'No subtrie under address: {address}')
        rest = address.rest()
        if not rest:
            return self.trie[key]
        return self.trie[key].get_subtrie(rest)

    def set_subtrie(self, address, subtrie):
        """Replace the entire subtrie at `address` with the given `subtrie`."""
        assert isinstance(subtrie, ChoiceTrie)
        # Primitive trie.
        if self.is_primitive():
            raise RuntimeError('Cannot set_subtrie of primitive MutableChoiceTrie.')
        # Compound trie.
        if not address:
            raise RuntimeError('Cannot set_subtrie at empty address.')
        key = address.first()
        rest = address.rest()
        if not rest:
            self.trie[key] = subtrie
            return None
        if key not in self.trie:
            self.trie[key] = MutableChoiceTrie()
        # Cannot recursively call set_subtrie on a primitive, so overwrite
        # the primitive with an empty trie.
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
        # Primitive trie.
        if self.is_primitive():
            return {addr(): self.trie[()]}
        # Compound trie.
        d = {}
        for k, subtrie in self.trie.items():
            subtrie_flat = subtrie.flatten()
            d_sub_prefix = {addr(k) + t: v for t, v in subtrie_flat.items()}
            d.update(d_sub_prefix)
        return d

    def asdict(self):
        # Primitive trie.
        if self.is_primitive():
            return dict(self.trie)
        # Compound trie.
        return {k: v.asdict() for k, v in self.trie.items()}

    def __getitem__(self, address):
        assert isinstance(address, ChoiceAddress)
        # Primitive trie.
        if self.is_primitive():
            return self.get_choice(address)
        # Compound trie.
        subtrie = self.get_subtrie(address)
        # Primitive subtrie: unbox
        if subtrie.is_primitive():
            return subtrie[addr()]
        # Compound subtrie.
        return subtrie

    def __setitem__(self, address, value):
        """Write `value` to the given `address`."""
        assert isinstance(address, ChoiceAddress)
        # Primitive trie.
        if self.is_primitive():
            # Cannot add new choices.
            if address:
                raise RuntimeError('Cannot add choices to a primitive MutableChoiceTrie.')
            # Overwrite the choice.
            self.trie[()] = value
            return None
        # Compound trie.
        if not address:
            if self.trie:
                raise RuntimeError('Cannot add choices to nonempty MutableChoiceTrie.')
            self.trie[()] = value
            return None
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
        for k, subtrie in self.trie.items():
            yield (addr(k), subtrie)

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
        # Primitive trie.
        if x.is_primitive():
            trie = MutableChoiceTrie()
            trie[addr()] = x[addr()]
            return trie
        # Compound trie.
        trie = MutableChoiceTrie()
        for address, subtrie in x:
            if subtrie.is_primitive():
                value = subtrie[addr()]
                trie[address] = value # deepcopy the value?
            else:
                subtrie_recursive = MutableChoiceTrie.copy(subtrie)
                trie.set_subtrie(address, subtrie_recursive)
        return trie
