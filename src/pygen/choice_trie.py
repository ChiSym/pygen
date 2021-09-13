from .choice_address import addr

class ChoiceTrie:
    """Implements a Trie (prefix tree) data structure.
    https://en.wikipedia.org/wiki/Trie

    The keys in the trie tree are instances of ChoiceAddress.
    The leaves (choices) in the trie are arbitrary Python values.

    Every trie is either (i) 'empty', (ii) 'primitive', or (iii) 'compound'.
    A primitive trie has a value; a compound has no value but has at least one subtrie;
    an empty trie has neither a value or any subtries.

    The truth value of an empty trie is `False` and the truth value of
    a primitive or compound trie is `True` (even when there are no values anywhere in the trie)
    """

    def has_choice(self):
        """Return True if and only if this trie has exactly one choice,
        stored under the empty address `addr()`, and zero subtries."""
        raise NotImplementedError()

    def get_choice(self):
        """Return the value stored under the empty address `addr()`."""
        raise NotImplementedError()

    def get_subtrie(self, address, strict=True):
        """Return the ChoiceTrie at the given `address`.

        If not `strict`, then fall back to returning an empty choice trie, which
        does not share data with `self` (so mutating it afterwards will not mutate `self`).

        For the empty address `addr()` return `self`.
        """
        raise NotImplementedError()

    def __getitem__(self, address):
        """Return the value of the choice at the given `address`."""
        raise NotImplementedError()

    def subtries(self):
        """Return an iterator over `(k, trie)` pairs that satisfy
        `self.get_subtrie(k)= trie`, excluding k `addr()` (empty address)."""
        raise NotImplementedError()

    def get_shallow_choices(self):
        # TODO document
        for k, subtrie in self.subtries():
            if subtrie.has_choice():
                yield (k, subtrie.get_choice())

    def get_shallow_subtries(self):
        # TODO document
        for k, subtrie in self.subtries():
            if not subtrie.has_choice():
                yield (k, subtrie)

    def asdict(self):
        # TODO document
        # Primitive trie.
        if self.has_choice():
            return {(): self.get_choice()}
        # Compound trie.
        return {k: v.asdict() for k, v in self.subtries()}


class MutableChoiceTrieError(Exception):
    pass


MCTError = MutableChoiceTrieError


class MutableChoiceTrie(ChoiceTrie):

    def __init__(self):
        self.trie = {}

    def has_choice(self):
        b = () in self.trie
        assert not b or len(self.trie) == 1
        return b

    def get_choice(self):
        try:
            return self.trie[()]
        except KeyError:
            raise MCTError('Cannot get_choice of a choice trie that is not primitive.')

    def set_value(self, value):
        self.trie = {(): value}

    def update(self, other):
        """Update this choice trie with the contents of the other; where the other takes precedence"""
        assert isinstance(other, ChoiceTrie)
        if not other:
            return
        elif other.has_choice() or self.has_choice():
            self.trie = MutableChoiceTrie.copy(other).trie
        else:
            for (k, other_subtrie) in other.subtries():
                self_subtrie = self.get_subtrie(addr(k), strict=False)
                self_subtrie.update(other_subtrie)
                self.set_subtrie(addr(k), self_subtrie)

    def get_subtrie(self, address, strict=True):
        if address:
            if self.has_choice():
                if strict:
                    raise MCTError(f'No subtrie at address {address}')
                return MutableChoiceTrie()
            else:
                key = address.first()
                rest = address.rest()
                if key in self.trie:
                    return self.trie[key].get_subtrie(rest, strict=strict)
                else:
                    if strict:
                        raise MCTError(f'No subtrie at address {address}')
                    return MutableChoiceTrie()
        else:
            return self

    def set_subtrie(self, address, subtrie):
        """Set subtrie at address"""
        if not isinstance(subtrie, ChoiceTrie):
            raise MCTError('Can only set subtrie to a ChoiceTrie value')
        if address:
            if self.has_choice():
                del self.trie[()]
            key = address.first()
            rest = address.rest()
            if key in self.trie:
                self.trie[key].set_subtrie(rest, subtrie)
            else:
                self.trie[key] = MutableChoiceTrie()
                self.trie[key].set_subtrie(rest, subtrie)
        else:
            self.trie = MutableChoiceTrie.copy(subtrie).trie

    def subtries(self):
        """Iterate over the children subtries and associated keys."""
        for k, subtrie in self.trie.items():
            if k != ():
                yield (k, subtrie)

    def __getitem__(self, address):
        """Return a the value of the choice at `address`."""
        return self.get_subtrie(address).get_choice()

    def __setitem__(self, address, value):
        """Set value of random choice at `address`."""
        subtrie = self.get_subtrie(address, strict=False)
        subtrie.set_value(value)
        self.set_subtrie(address, subtrie)

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
        if x.has_choice():
            trie = MutableChoiceTrie()
            trie.set_value(x.get_choice())
            return trie
        # Compound trie.
        trie = MutableChoiceTrie()
        for k, subtrie in x.subtries():
            subtrie_recursive = MutableChoiceTrie.copy(subtrie)
            trie.set_subtrie(addr(k), subtrie_recursive)
        return trie


def empty_trie_or_error(error_msg, strict):
    if strict is None or strict:
        raise MCTError(error_msg)
    return MutableChoiceTrie()
