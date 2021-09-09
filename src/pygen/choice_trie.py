from .choice_address import ChoiceAddress
from .choice_address import addr

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
