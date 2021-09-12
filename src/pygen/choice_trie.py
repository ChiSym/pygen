from .choice_address import ChoiceAddress
from .choice_address import addr


def choice_trie():
    trie = MutableChoiceTrie()
    return trie.flat_view()

# NOTE: the first element of an address cannot be ()


class ChoiceTrieFlatView:
    """A view of a choice trie as an associative array
    mapping addresses of random choices to values."""

    def __init__(self, hierarchical_view):
        assert isinstance(hierarchical_view, ChoiceTrie)
        self.hierarchical_view = hierarchical_view

    def hierarchical_view(self):
        """Return a hierarchical view of the choice trie."""
        return self.hierarchical_view

    def __getitem__(self, address):
        """Return a the value of the choice at `address`."""
        subtrie = self.hierarchical_view[address]
        return subtrie.get_value()

    @staticmethod
    def _flatten(hierarchical_view):
        # Primitive trie.
        if hierarchical_view.is_primitive():
            return {addr(): hierarchical_view.get_value()}
        # Compound trie.
        d = {}
        for k, subtrie in hierarchical_view:
            subtrie_flat = ChoiceTrieFlatView._flatten(subtrie)
            d_sub_prefix = {addr(k) + t: v for t, v in subtrie_flat.items()}
            d.update(d_sub_prefix)
        return d

    def __iter__(self):
        """Iterate over the (address, value) pairs for all random choices."""
        return iter(ChoiceTrieFlatView._flatten(self.hierarchical_view).items())

    def __str__(self):
        return str({k: v for (k, v) in self})




class ChoiceTrie:
    """Implements a Trie (prefix tree) data structure.
    https://en.wikipedia.org/wiki/Trie

    The keys in the trie tree are instances of ChoiceAddress.
    The leaves (choices) in the trie are arbitrary Python values.

    Every trie is either (i) 'empty', (ii) 'primitive', or (iii) 'compound'.
    A primitive trie has a choice at the empty address (`addr()`) and
    no other internal nodes. A compound has at least one internal node
    and no choice at the empty address.

    The truth value of an empty trie is `False` and the truth value of
    a primitive or compound trie is `True` (even when there are no leaves in
    the compound trie).
    """

    # TODO: rename to has_value()
    def is_primitive(self):
        """Return True if and only if this trie has exactly one choice,
        stored under the empty address `addr()`."""
        raise NotImplementedError()

    def get_value(self):
        raise NotImplementedError()

    def flat_view(self):
        """Return a `ChoiceTrieFlatView` of this choice trie."""
        raise NotImplementedError()

    def get(self, address, strict=True):
        """Return the ChoiceTrie at the given `address`, or an empty choice trie."""
        raise NotImplementedError()

    def __getitem__(self, address):
        """Return the ChoiceTrie at the given `address`.

        If `address` is the empty address `addr()` then return self.
        """
        raise NotImplementedError()

    def __iter__(self):
        """Return an iterator over `(k, trie)` pairs that satisfy `self[k] = trie`."""
        raise NotImplementedError()


class MutableChoiceTrieError(Exception):
    pass


MCTError = MutableChoiceTrieError

class MutableChoiceTrieFlatView(ChoiceTrieFlatView):

    def __setitem__(self, address, value):
        subtrie = self.hierarchical_view.get(address, strict=False)
        subtrie.trie = {(): value}
        self.hierarchical_view[address] = subtrie



class MutableChoiceTrie(ChoiceTrie):

    def __init__(self):
        self.trie = {}

    def is_primitive(self):
        b = () in self.trie
        assert not b or len(self.trie) == 1
        return b

    def get_value(self):
        try:
            return self.trie[()]
        except KeyError:
            raise MCTError('Cannot get_value of a choice trie that is not primitive.')

    def set_value(self, value):
        self.trie = {(): value}

    def flat_view(self):
        return MutableChoiceTrieFlatView(self)

    def update(self, other):
        """Update this choice trie with the contents of the other; where the other takes precedence"""
        assert isinstance(other, ChoiceTrie)
        if not other:
            return
        elif other.is_primitive() or self.is_primitive():
            self.trie = MutableChoiceTrie.copy(other).trie
        else:
            for (k, other_subtrie) in other:
                self_subtrie = self.get(addr(k), strict=False)
                self_subtrie.update(other_subtrie)
                self[addr(k)] = self_subtrie

    def get_shallow_choices(self):
        for k, subtrie in self:
            if subtrie.is_primitive():
                yield (k, subtrie[addr()])

    def get_shallow_subtries(self):
        for k, subtrie in self:
            if not subtrie.is_primitive():
                yield (k, subtrie)

    def asdict(self):
        # Primitive trie.
        if self.is_primitive():
            return dict(self.trie)
        # Compound trie.
        return {k: v.asdict() for k, v in self.trie.items()}

    def get(self, address, strict=True):
        if address:
            if self.is_primitive():
                if strict:
                    raise MCTError(f'No subtrie at address {address}')
                else:
                    return MutableChoiceTrie()
            else:
                key = address.first()
                rest = address.rest()
                if key in self.trie:
                    return self.trie[key].get(rest, strict=strict)
                else:
                    if strict:
                        raise MCTError(f'No subtrie at address {address}')
                    else:
                        return MutableChoiceTrie()
        else:
            return self

    def __getitem__(self, address):
        return self.get(address, strict=True)

    def __setitem__(self, address, subtrie):
        if not isinstance(subtrie, ChoiceTrie):
            raise MCTError('Can only set subtrie to a ChoiceTrie value')
        if address:
            if self.is_primitive():
                del self.trie[()]
            key = address.first()
            rest = address.rest()
            if key in self.trie:
                self.trie[key][rest] = subtrie
            else:
                self.trie[key] = MutableChoiceTrie()
                self.trie[key][rest] = subtrie
        else:
            self.trie = MutableChoiceTrie.copy(subtrie).trie

    def __iter__(self):
        for k, subtrie in self.trie.items():
            if k != ():
                yield (k, subtrie)

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
            trie.set_value(x.get_value())
            return trie
        # Compound trie.
        trie = MutableChoiceTrie()
        for k, subtrie in x:
            subtrie_recursive = MutableChoiceTrie.copy(subtrie)
            trie[addr(k)] = subtrie_recursive
            # if subtrie.is_primitive():
            #     value = subtrie.get_value()
            #     trie[address] = value # deepcopy the value?
            # else:
            #     subtrie_recursive = MutableChoiceTrie.copy(subtrie)
            #     trie[address] = subtrie_recursive
        return trie

def empty_trie_or_error(error_msg, strict):
    if strict is None or strict:
        raise MCTError(error_msg)
    return MutableChoiceTrie()
