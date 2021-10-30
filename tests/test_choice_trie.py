import pytest

from pygen.choice_address import addr
from pygen.choice_trie import ChoiceTrie
from pygen.choice_trie import MutableChoiceTrie
from pygen.choice_trie import MutableChoiceTrieError


def test_trie_empty():
    trie = MutableChoiceTrie()
    # Cannot iterate directly.
    with pytest.raises(NotImplementedError):
        for (k, subtrie) in trie:
            assert False
    # Various properties of an empty trie.
    assert not trie.has_choice()
    assert trie.asdict() == {}
    assert MutableChoiceTrie.copy(trie) == trie
    assert MutableChoiceTrie.copy(trie) is not trie
    assert trie.has_subtrie(addr())
    assert trie.get_subtrie(addr()) == trie
    with pytest.raises(MutableChoiceTrieError):
        trie.get_choice()
    with pytest.raises(MutableChoiceTrieError):
        trie[addr()]
    with pytest.raises(MutableChoiceTrieError):
        trie[addr('a')]  # there is no choice at 'a'
    with pytest.raises(MutableChoiceTrieError):
        trie.get_subtrie(addr('a'))  # there is no subtrie at 'a'
    assert trie.get_subtrie(addr('a'), strict=False) == MutableChoiceTrie()
    # Iteration.
    for k, subtrie in trie.choices():
        assert False
    for k, subtrie in trie.subtries():
        assert False

def test_trie_primitive():
    trie = MutableChoiceTrie()
    trie[addr()] = 'h'
    assert trie[addr()] == trie.get_choice() == 'h'
    assert trie.has_choice()
    assert trie.asdict() == {(): 'h'}
    # Copying gives an identical trie.
    assert MutableChoiceTrie.copy(trie) == trie
    assert MutableChoiceTrie.copy(trie) is not trie
    # Can overwrite primitive trie
    trie[addr()] = 2
    assert trie[addr()] == trie.get_choice() == 2
    assert trie.asdict() == {(): 2}
    # Subtrie of primitive at empty address is the same trie
    assert trie.has_subtrie(addr())
    assert trie.get_subtrie(addr()) is trie
    assert not trie.has_subtrie(addr('a'))
    # Check for no choices.
    with pytest.raises(MutableChoiceTrieError):
        trie[addr('z')]
    # Cannot get subtrie of primitive at non-empty address if strict
    with pytest.raises(MutableChoiceTrieError):
        trie.get_subtrie(addr('z'))
    subtrie = MutableChoiceTrie()
    assert trie.get_subtrie(addr('z'), strict=False) == subtrie
    # Set subtrie at non-empty address.
    trie.set_subtrie(addr('z'), subtrie)
    assert not trie.has_choice()
    # Set subtrie of primitive at empty address.
    trie.set_subtrie(addr(), MutableChoiceTrie())
    assert not trie
    # Iteration.
    for k, v in trie.subtries():
        assert False
    for address, v in trie.choices():
        assert address == addr()
        assert v == 2

def test_trie_single_address():
    trie = MutableChoiceTrie()
    trie[addr('a')] = 1
    assert trie[addr('a')] == 1
    assert not trie.has_choice()
    with pytest.raises(MutableChoiceTrieError):
        trie.get_choice()
    with pytest.raises(MutableChoiceTrieError):
        trie[addr()]
    with pytest.raises(MutableChoiceTrieError):
        trie[addr('b')]
    assert trie.has_subtrie(addr('a'))
    assert trie.has_subtrie(addr())
    assert trie.get_subtrie(addr()) is trie
    assert trie.asdict() == {'a': {(): 1}}
    subtrie = trie.get_subtrie(addr('a'))
    assert subtrie.has_choice()
    assert subtrie[addr()] == subtrie.get_choice() == 1
    # Copying gives an identical trie.
    assert MutableChoiceTrie.copy(trie) == trie
    assert MutableChoiceTrie.copy(trie) is not trie
    # Iteration.
    for k, v in trie.subtries():
        assert k == addr('a')
        assert v.has_choice()
        assert v.get_choice() == 1
    assert len(list(trie.subtries())) == 1
    for k, v in trie.choices():
        assert k == addr('a')
        assert v == 1
    assert len(list(trie.choices())) == 1

def test_trie_set_get_empty_address():
    trie = MutableChoiceTrie()
    assert trie.get_subtrie(addr()) is trie
    assert trie.get_subtrie(addr(), strict=False) is trie
    trie[addr('a')] = 1
    assert trie.get_subtrie(addr()) is trie
    assert trie.get_subtrie(addr(), strict=False) is trie
    # Setting subtrie at empty address overwrites trie.
    trie.set_subtrie(addr(), MutableChoiceTrie())
    assert not trie

def test_trie_tuples_as_keys():
    # Trie with tuples as the keys.
    for k in [('a',), (('a', 'b'),)]:
        trie = MutableChoiceTrie()
        trie[addr(k)] = 10
        assert trie[addr(k)] == 10
        assert trie.asdict() == {k: {(): 10}}
        assert not trie.has_choice()
        subtrie = trie.get_subtrie(addr(k))
        assert subtrie.has_choice()

def test_trie_ovewrite_primitive_primitive():
    trie = MutableChoiceTrie()
    trie[addr('a')] = 2
    assert trie[addr('a')] == 2
    trie[addr('a')] = 3
    assert trie[addr('a')] == 3

def test_trie_ovewrite_primitive_nonprimitive():
    trie = MutableChoiceTrie()
    trie[addr('a')] = 2
    assert trie.asdict() == {'a': {(): 2}}
    trie[addr('a', 'b')] = 2
    assert trie.asdict() == {'a': {'b': {(): 2}}}
    subtrie = MutableChoiceTrie()
    subtrie[addr('c')] = 10
    trie.set_subtrie(addr('a','b'), subtrie)
    assert trie.asdict() == {'a': {'b': {'c': {(): 10}}}}
    trie[addr('a','x')] = 1
    assert trie.asdict() == {'a': {'b': {'c': {(): 10}}, 'x': {(): 1}}}
    trie[addr('a')] = 1
    assert trie.get_subtrie(addr('a')).has_choice()
    assert trie[addr('a')] == 1

def test_trie_interactive_session_1():
    trie = MutableChoiceTrie()
    # Create a primitive.
    trie[addr('a')] = 2
    assert trie.asdict() == {'a': {(): 2}}
    # Create a compound.
    trie[addr('b', 'c')] = 3
    assert trie.asdict() == {'a': {(): 2}, 'b': {'c': {(): 3}}}
    subtrie = trie.get_subtrie(addr('b'))
    assert not subtrie.has_choice()
    # Test difference between get_choice and __getitem__
    assert not trie.has_choice()
    with pytest.raises(MutableChoiceTrieError):
        trie.get_choice()
    assert isinstance(trie.get_subtrie(addr('b')), ChoiceTrie)
    subtrie = trie.get_subtrie(addr('b', 'c'))
    assert subtrie.has_choice()
    assert subtrie[addr()] == 3
    assert subtrie.get_choice() == 3
    # Extend a compound.
    trie[addr('b', 'd')] = 14
    assert trie.asdict() == {'a': {(): 2}, 'b': {'c': {(): 3}, 'd': {(): 14}}}
    with pytest.raises(MutableChoiceTrieError):
        trie[addr('b')]
    assert trie[addr('b', 'd')] == 14
    assert trie.get_subtrie(addr('b')).asdict() == {'c': {(): 3}, 'd': {(): 14}}
    with pytest.raises(MutableChoiceTrieError):
        trie.get_subtrie(addr('d'))
    subtrie = trie.get_subtrie(addr('b', 'd'))
    assert subtrie.has_choice()
    assert subtrie[addr()] == 14
    # Overwrite a primitive with a compound.
    trie[addr('a', 'c')] = 5
    subtrie = trie.get_subtrie(addr('a'))
    assert subtrie is trie.get_subtrie(addr('a'))
    assert not subtrie.has_choice()
    assert subtrie[addr('c')] == 5
    # Confirm values.
    assert trie[addr('a', 'c')] == 5
    assert trie[addr('b', 'c')] == 3
    assert trie[addr('b', 'd')] == 14
    # Overwrite a compound.
    trie[addr('b', 'c')] = 13
    assert trie[addr('b', 'c')] == 13
    # Add a few more address.
    trie[addr(1)] = 5
    trie.set_subtrie(addr('a', 'f'), MutableChoiceTrie())
    # Test iteration.
    assert trie.asdict() == {
        'a': {'c': {(): 5}, 'f': {}},
        'b': {'c': {(): 13}, 'd': {(): 14}},
        1: {(): 5}
    }
    expected = [addr('a'), addr('b'), addr(1)]
    for k, v in trie.subtries():
        assert k in expected
        expected = [e for e in expected if e != k]
    assert not expected
    expected = [addr('a','c'), addr('b','c'), addr('b','d'), addr(1)]
    for k, v in trie.choices():
        assert k in expected
        expected = [e for e in expected if e != k]
    assert not expected

def test_trie_primitive_ChoiceTrie_vs_primitive_dict():
    # Writing a dict versus ChoiceTrie as primitive choice.
    # = Write a dict.
    trie_value = {'a' : {(): 1.123}}
    trie = MutableChoiceTrie()
    trie[addr('b')] =  trie_value
    assert trie.get_subtrie(addr('b')).has_choice()
    assert trie[addr('b')] == trie_value
    d1 = trie.asdict()
    assert d1 == {'b': {(): trie_value}}
    # = Write a MutableChoiceTrie.
    trie_value = MutableChoiceTrie()
    trie_value[addr('a')] = 1.123
    trie = MutableChoiceTrie()
    trie[addr('b')] = trie_value
    assert trie.get_subtrie(addr('b')).has_choice()
    assert trie[addr('b')] == trie_value
    assert trie.get_subtrie(addr('b'))[addr()] == trie_value
    d2 = trie.asdict()
    assert d2 == {'b': {(): trie_value}}
    # = Confirm they are not equal
    assert d1 != d2

def test_subtrie_nested():
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
    assert set(trie.choices()) == {
        (addr('a'), 1.123),
        (addr('b', 'c'), 10),
        (addr('b', 'e', 1), 11)
    }
    assert MutableChoiceTrie.copy(trie) == trie

def test_trie_primitive_ChoiceTrie_vs_subtrie():
    # Setting a subtrie versus setting a primitive using index.
    # = setting a primitive that happens to be a choice trie
    trie = MutableChoiceTrie()
    trie[addr('a')] = MutableChoiceTrie()
    assert trie.asdict() == {'a': {(): MutableChoiceTrie()}}
    assert dict(trie.choices()) == {addr('a'): trie[addr('a')]}
    assert dict(trie.subtries()) == {addr('a'): trie.get_subtrie(addr('a'))}
    # = setting a subtrie proper
    trie.set_subtrie(addr('a'), MutableChoiceTrie())
    assert trie.trie == {'a': MutableChoiceTrie()}
    assert trie.asdict() == {'a': {}}
    assert dict(trie.choices()) == {}

def test_flatten_non_empty_subtrie_with_no_choices():
    # Flatten a non-empty subtrie with no choices.
    trie = MutableChoiceTrie()
    trie.set_subtrie(addr('a'), MutableChoiceTrie())
    assert list(trie.choices()) == []

def test_subtrie_mutation_propagates():
    trie = MutableChoiceTrie()
    trie.set_subtrie(addr('a'), MutableChoiceTrie())
    subtrie = trie.get_subtrie(addr('a'))
    subtrie[addr('b')] = 1.123
    assert trie[addr('a', 'b')] == 1.123

def test_subtrie_circular_basic():
    trie = MutableChoiceTrie()
    trie[addr()] = trie
    assert trie.has_choice()
    assert trie[addr()] == trie

    # Should really be forbidden.
    # The mutation semantics are confusing.
    # Please do not set a trie to include itself.
    trie = MutableChoiceTrie()
    trie.set_subtrie(addr('a'), trie)
    assert trie.asdict() == {'a': {'a': {}}}
    assert trie.get_subtrie(addr('a')) is not trie
    trie.set_subtrie(addr('b'), MutableChoiceTrie())
    trie[addr('c')] = 1
    assert trie.asdict() == {'a': {'a': {}}, 'b': {}, 'c': {(): 1}}

def test_subtrie_circular_complex():
    trie = MutableChoiceTrie()
    trie[addr('a')] = 1
    subtrie = MutableChoiceTrie()
    subtrie[addr('c')] = 'foo'
    trie.set_subtrie(addr('a'), subtrie)
    assert trie.get_subtrie(addr('a')) == subtrie
    trie.set_subtrie(addr('a', 'b'), subtrie)
    assert trie.get_subtrie(addr('a', 'b')) == subtrie

def test_update_empty_or_primitive():
    # other is primitive
    other = MutableChoiceTrie()
    other[addr()] = 2
    trie = MutableChoiceTrie()
    trie[addr()] = 1
    trie.update(other)
    assert trie == other
    trie = MutableChoiceTrie()
    trie.update(other)
    assert trie == other

    # other is empty
    other = MutableChoiceTrie()
    trie = MutableChoiceTrie()
    trie.update(other)
    assert trie == other
    trie = MutableChoiceTrie()
    trie[addr()] = 1
    trie.update(other)
    assert trie[addr()] == 1

    # other is not empty or primitive
    other = MutableChoiceTrie()
    other[addr('a')] = 2
    inner = MutableChoiceTrie()
    inner[addr('c')] = 3
    other.set_subtrie(addr('b'), inner)
    trie = MutableChoiceTrie()
    trie.update(other)
    assert trie == other
    trie = MutableChoiceTrie()
    trie[addr()] = 1
    trie.update(other)
    assert trie == other
    trie[addr('a')] = 5
    assert other[addr('a')] == 2

    # self is primitive
    trie = MutableChoiceTrie()
    trie[addr('a')] = 1
    other = MutableChoiceTrie()
    other[addr('a', 'b')] = 2
    trie.update(other)
    trie[addr('a', 'b')] = 7
    assert other[addr('a', 'b')] == 2

def test_update_nonprimitive():

    def make_original():
        trie = MutableChoiceTrie()
        trie[addr('a')] = 1
        trie[addr('b')] = 2
        inner = MutableChoiceTrie()
        inner[addr('d')] = 3
        trie.set_subtrie(addr('c'), inner)
        return trie

    # other is primitive
    other = MutableChoiceTrie()
    other[addr()] = 2
    trie = make_original()
    trie.update(other)
    assert trie == other
    other[addr()] = 3
    assert trie[addr()] == 2

    # other is empty
    other = MutableChoiceTrie()
    trie = make_original()
    trie.update(other)
    assert trie == make_original()

    # other is not empty or primitive
    other = MutableChoiceTrie()
    other[addr('a')] = 4
    inner = MutableChoiceTrie()
    inner[addr('e', 'f')] = 5
    other.set_subtrie(addr('c'), inner)
    trie = make_original()
    trie.update(other)
    expected = MutableChoiceTrie()
    expected[addr('a')] = 4
    expected[addr('b')] = 2
    expected[addr('c', 'd')] = 3
    expected[addr('c', 'e', 'f')] = 5
    assert trie == expected
    other[addr('c', 'e', 'f')] = 1
    assert trie == expected
