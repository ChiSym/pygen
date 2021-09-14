class ChoiceTrie:
    """Implements a Trie (prefix tree) data structure.
    https://en.wikipedia.org/wiki/Trie

    A path in the trie is specified by a Python tuple.
    The leaves (choices) in the trie are arbitrary Python values.

    Every trie is either (i) 'empty', (ii) 'primitive', or (iii) 'compound'.
    A primitive trie has a choice; a compound has no choice and at least one subtrie;
    an empty trie has neither a choice or any subtries.

    The truth value of an empty trie is `False` and the truth value of
    a primitive or compound trie is `True` (even when there are no values anywhere in the trie)
    """

    def has_choice(self):
        raise NotImplementedError()

    def get_choice(self):
        raise NotImplementedError()

    def has_subtrie(self, address):
        raise NotImplementedError()

    def get_subtrie(self, address, strict=None):
        raise NotImplementedError()

    def __getitem__(self, address):
        raise NotImplementedError()

    def subtries(self):
        raise NotImplementedError()

    def choices(self):
        raise NotImplementedError()

    # Convenience methods.
    # Determine whether to add to formal API.

    def choices_shallow(self):
        for address, subtrie in self.subtries():
            if subtrie.has_choice():
                yield (address, subtrie.get_choice())

    def subtries_shallow(self):
        for address, subtrie in self.subtries():
            if not subtrie.has_choice():
                yield (address, subtrie)

    def asdict(self):
        # Primitive.
        if self.has_choice():
            return {(): self.get_choice()}
        # Compound.
        return {k[0]: v.asdict() for k, v in self.subtries()}


class MutableChoiceTrieError(Exception):
    pass

MCTError = MutableChoiceTrieError

tupleify = lambda x: x if isinstance(x, tuple) else (x,)

class MutableChoiceTrie(ChoiceTrie):

    # Core API.

    def __init__(self):
        self.trie = {}

    def has_choice(self):
        b = () in self.trie
        assert not b or len(self.trie) == 1
        return b

    def get_choice(self):
        if () not in self.trie:
            raise MCTError('Cannot get_choice of a ChoiceTrie that has no choice.')
        assert len(self.trie) == 1
        return self.trie[()]

    def has_subtrie(self, address):
        if not address:
            return True
        # Primitive.
        if self.has_choice():
            return False
        # Compound.
        try:
            self.get_subtrie(address)
            return True
        except MCTError:
            return False

    def get_subtrie(self, address, strict=None):
        if not address:
            return self
        # Primitive.
        if self.has_choice():
            if strict is None or strict:
                raise MCTError(f'No subtrie at address {address}')
            return MutableChoiceTrie()
        # Compound.
        address = tupleify(address)
        key = address[0]
        rest = address[1:]
        if key in self.trie:
            return self.trie[key].get_subtrie(rest, strict=strict)
        if strict is None or strict:
            raise MCTError(f'No subtrie at address {address}')
        return MutableChoiceTrie()

    def __getitem__(self, address):
        return self.get_subtrie(address).get_choice()

    def subtries(self):
        if not self.has_choice():
            for k, subtrie in self.trie.items():
                # Return (k,) for now, until
                # there exists syntactic sugar
                # for handling non-addr keys
                # in the API.
                yield ((k,), subtrie)

    def choices(self):
        if self.has_choice():
            yield ((), self.get_choice())
        for address, subtrie in self.subtries():
            for subaddress, choice in subtrie.choices():
                # TODO: Make API for adding addresses.
                path = address + subaddress
                yield path, choice

    def __bool__(self):
        return bool(self.trie)

    def __iter__(self):
        raise NotImplementedError('Cannot iterate over trie, '
            'use trie.choices() or trie.subtries().')

    # MutableChoiceTrie API.

    def set_subtrie(self, address, subtrie):
        if not isinstance(subtrie, ChoiceTrie):
            raise MCTError('Can only set subtrie to a ChoiceTrie.')
        # Replacing entire trie.
        if not address:
            self.trie = MutableChoiceTrie.copy(subtrie).trie
            return
        # Modifying subtrie.
        if self.has_choice():
            del self.trie[()]
        address = tupleify(address)
        key = address[0]
        rest = address[1:]
        if key not in self.trie:
            self.trie[key] = MutableChoiceTrie()
        self.trie[key].set_subtrie(rest, subtrie)

    def set_choice(self, choice):
        self.trie = {(): choice}

    def __setitem__(self, address, choice):
        subtrie = self.get_subtrie(address, strict=False)
        subtrie.set_choice(choice)
        self.set_subtrie(address, subtrie)

    def update(self, other):
        assert isinstance(other, ChoiceTrie)
        if not other:
            return
        if other.has_choice() or self.has_choice():
            self.trie = MutableChoiceTrie.copy(other).trie
        else:
            for (address, other_subtrie) in other.subtries():
                self_subtrie = self.get_subtrie(address, strict=False)
                self_subtrie.update(other_subtrie)
                self.set_subtrie(address, self_subtrie)

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
        if x.has_choice():
            trie = MutableChoiceTrie()
            trie.set_choice(x.get_choice())
            return trie
        # Compound trie.
        trie = MutableChoiceTrie()
        for address, subtrie in x.subtries():
            subtrie_recursive = MutableChoiceTrie.copy(subtrie)
            trie.set_subtrie(address, subtrie_recursive)
        return trie
