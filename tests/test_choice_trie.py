import pytest

from pygen.choice_address import addr
from pygen.choice_trie import MutableChoiceTrie

def test_trie_empty():
    # Empty trie.
    trie = MutableChoiceTrie()
    assert not trie.is_primitive()
    assert trie.flatten() == {}
    assert trie.asdict() == {}
    assert MutableChoiceTrie.copy(trie) == trie
    assert MutableChoiceTrie.copy(trie) is not trie

def test_trie_primitive():
    # Primitive trie.
    trie = MutableChoiceTrie()
    trie[addr()] = 1
    assert trie.is_primitive()
    assert trie.asdict() == {(): 1}
    # Flattening gives a single entry.
    assert trie.flatten() == {addr(): 1}
    # Copying gives an identical trie.
    assert MutableChoiceTrie.copy(trie) == trie
    assert MutableChoiceTrie.copy(trie) is not trie
    # Cannot write to a primitive trie.
    with pytest.raises(RuntimeError):
        trie[addr('a')] = 1
    # Can overwrite primitive trie.
    trie[addr()] = 2
    assert trie[addr()] == 2
    assert trie.asdict() == {(): 2}
    # Cannot get subtrie of primitive.
    with pytest.raises(RuntimeError):
        trie.get_subtrie(addr())
    # Cannot set subtrie of primitive.
    with pytest.raises(RuntimeError):
        trie.set_subtrie(addr(), MutableChoiceTrie())

def test_trie_single_address():
    # Trie with single address.
    trie = MutableChoiceTrie()
    trie[addr('a')] = 1
    assert trie[addr('a')] == 1
    assert not trie.is_primitive()
    with pytest.raises(RuntimeError):
        trie[addr()]
    assert trie.asdict() == {'a': {(): 1}}
    subtrie = trie.get_subtrie(addr('a'))
    assert subtrie.is_primitive()
    assert subtrie[addr()] == 1

def test_trie_set_get_empty_address():
    trie = MutableChoiceTrie()
    trie[addr('a')] = 1
    with pytest.raises(RuntimeError):
        trie.get_subtrie(addr())
    with pytest.raises(RuntimeError):
        trie.set_subtrie(addr(), MutableChoiceTrie())

def test_trie_tuples_as_keys():
    # Trie with tuples as the keys.
    for k in [('a',), (('a', 'b'),)]:
        trie = MutableChoiceTrie()
        trie[addr(k)] = 10
        assert trie.asdict() == {k: {(): 10}}
        assert not trie.is_primitive()
        subtrie = trie.get_subtrie(addr(k))
        assert subtrie.is_primitive()

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

def test_trie_interactive_session_1():
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
    with pytest.raises(RuntimeError):
        trie.get_subtrie(addr('d'))
    subtrie = trie.get_subtrie(addr('b', 'd'))
    assert subtrie.is_primitive()
    assert subtrie[addr()] == 14
    # Overwrite a primitive with a compound.
    trie[addr('a', 'c')] = 5
    subtrie = trie.get_subtrie(addr('a'))
    assert not subtrie.is_primitive()
    assert subtrie == trie[addr('a')]
    assert subtrie[addr('c')] == 5
    # Confirm values.
    assert trie[addr('a', 'c')] == 5
    assert trie[addr('b', 'c')] == 3
    assert trie[addr('b', 'd')] == 14
    # Overwrite a compound.
    trie[addr('b', 'c')] = 13
    assert trie[addr('b', 'c')] == 13

def test_trie_primitive_ChoiceTrie_vs_primitive_dict():
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
    assert trie.flatten() == {
        addr('a'): 1.123,
        addr('b', 'c') : 10,
        addr('b', 'e', 1): 11
    }
    assert MutableChoiceTrie.copy(trie) == trie

def test_trie_primitive_ChoiceTrie_vs_subtrie():
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

def test_flatten_non_empty_subtrie_with_no_choices():
    # Flatten a non-empty subtrie with no choices.
    trie = MutableChoiceTrie()
    trie.set_subtrie(addr('a'), MutableChoiceTrie())
    assert trie.flatten() == {}

def test_subtrie_mutation_propagates():
    trie = MutableChoiceTrie()
    trie.set_subtrie(addr('a'), MutableChoiceTrie())
    subtrie = trie.get_subtrie(addr('a'))
    subtrie[addr('b')] = 1.123
    assert trie[addr('a', 'b')] == 1.123

def test_subtrie_circular_basic():
    # XXX Warning: Cannot print(trie), infinite recursion.
    trie = MutableChoiceTrie()
    trie.set_subtrie(addr('a'), trie)
    assert trie[addr('a')] == trie

def test_subtrie_circular_complex():
    trie = MutableChoiceTrie()
    trie[addr('a')] = 1
    subtrie = MutableChoiceTrie()
    subtrie[addr('c')] = 'foo'
    trie.set_subtrie(addr('a'), subtrie)
    assert trie.get_subtrie(addr('a')) == subtrie
    trie.set_subtrie(addr('a', 'b'), subtrie)
    assert trie.get_subtrie(addr('a', 'b')) == subtrie
    # XXX Warning: Cannot print(subtrie), infinite recursion.
