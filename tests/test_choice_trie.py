import pytest

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
    assert trie.has_subtrie(())
    assert trie.get_subtrie(()) == trie
    with pytest.raises(MutableChoiceTrieError):
        trie.get_choice()
    with pytest.raises(MutableChoiceTrieError):
        trie[()]
    with pytest.raises(MutableChoiceTrieError):
        trie[('a')]  # there is no choice at 'a'
    with pytest.raises(MutableChoiceTrieError):
        trie.get_subtrie(('a'))  # there is no subtrie at 'a'
    assert trie.get_subtrie(('a'), strict=False) == MutableChoiceTrie()
    # Iteration.
    for k, subtrie in trie.choices():
        assert False
    for k, subtrie in trie.subtries():
        assert False

def test_trie_primitive():
    trie = MutableChoiceTrie()
    trie[()] = 'h'
    assert trie[()] == trie.get_choice() == 'h'
    assert trie.has_choice()
    assert trie.asdict() == {(): 'h'}
    # Copying gives an identical trie.
    assert MutableChoiceTrie.copy(trie) == trie
    assert MutableChoiceTrie.copy(trie) is not trie
    # Can overwrite primitive trie
    trie[()] = 2
    assert trie[()] == trie.get_choice() == 2
    assert trie.asdict() == {(): 2}
    # Subtrie of primitive at empty address is the same trie
    assert trie.has_subtrie(())
    assert trie.get_subtrie(()) is trie
    assert not trie.has_subtrie(('a'))
    # Check for no choices.
    with pytest.raises(MutableChoiceTrieError):
        trie[('z')]
    # Cannot get subtrie of primitive at non-empty address if strict
    with pytest.raises(MutableChoiceTrieError):
        trie.get_subtrie(('z'))
    subtrie = MutableChoiceTrie()
    assert trie.get_subtrie(('z'), strict=False) == subtrie
    # Set subtrie at non-empty address.
    trie.set_subtrie(('z'), subtrie)
    assert not trie.has_choice()
    # Set subtrie of primitive at empty address.
    trie.set_subtrie((), MutableChoiceTrie())
    assert not trie
    # Iteration.
    for k, v in trie.subtries():
        assert False
    for address, v in trie.choices():
        assert address == ()
        assert v == 2

def test_trie_single_address():
    trie = MutableChoiceTrie()
    trie[('a')] = 1
    assert trie[('a')] == trie['a'] == trie['a',] == trie[('a',)] == 1
    assert not trie.has_choice()
    with pytest.raises(MutableChoiceTrieError):
        trie[('a',),]
    with pytest.raises(MutableChoiceTrieError):
        trie.get_choice()
    with pytest.raises(MutableChoiceTrieError):
        trie[()]
    with pytest.raises(MutableChoiceTrieError):
        trie[('b')]
    assert trie.has_subtrie(('a'))
    assert trie.has_subtrie(())
    assert trie.get_subtrie(()) is trie
    assert trie.asdict() == {'a': {(): 1}}
    subtrie = trie.get_subtrie(('a'))
    assert subtrie.has_choice()
    assert subtrie[()] == subtrie.get_choice() == 1
    # Copying gives an identical trie.
    assert MutableChoiceTrie.copy(trie) == trie
    assert MutableChoiceTrie.copy(trie) is not trie
    # Iteration.
    for k, v in trie.subtries():
        assert k == ('a',)
        assert v.has_choice()
        assert v.get_choice() == 1
    assert len(list(trie.subtries())) == 1
    for k, v in trie.choices():
        assert k == ('a',)
        assert v == 1
    assert len(list(trie.choices())) == 1

def test_trie_set_get_empty_address():
    trie = MutableChoiceTrie()
    assert trie.get_subtrie(()) is trie
    assert trie.get_subtrie((), strict=False) is trie
    trie[('a')] = 1
    assert trie.get_subtrie(()) is trie
    assert trie.get_subtrie((), strict=False) is trie
    # Setting subtrie at empty address overwrites trie.
    trie.set_subtrie((), MutableChoiceTrie())
    assert not trie

def test_trie_tuples_as_keys():
    # Trie with tuples as the keys.
    k = ('a',)
    trie = MutableChoiceTrie()
    trie[k] = 10
    assert trie[k] == 10
    assert trie.asdict() == {'a': {(): 10}}
    assert trie.has_subtrie(k)
    assert trie.has_subtrie(k[0])
    assert not trie.has_choice()
    subtrie = trie.get_subtrie(k)
    assert subtrie.has_choice()
    subtrie = trie.get_subtrie(k[0])
    assert subtrie.has_choice()

    k = (('a',),)
    trie = MutableChoiceTrie()
    trie[k] = 10
    assert trie[k] == 10
    assert trie.asdict() == {('a',): {(): 10}}
    assert trie.has_subtrie(k)
    assert not trie.has_subtrie(k[0])
    assert not trie.has_choice()
    subtrie = trie.get_subtrie(k)
    assert subtrie.has_choice()

    k = ('a', 'b')
    trie = MutableChoiceTrie()
    trie[(k,)] = 10
    assert trie[k,] == 10
    assert trie.asdict() == {('a', 'b'): {(): 10}}
    assert not trie.has_choice()
    subtrie = trie.get_subtrie((k,))
    assert subtrie.has_choice()

    assert not trie.has_subtrie(k)
    assert trie.has_subtrie((k,))

def test_trie_ovewrite_primitive_primitive():
    trie = MutableChoiceTrie()
    trie[('a')] = 2
    assert trie[('a')] == 2
    trie[('a')] = 3
    assert trie[('a')] == 3

def test_trie_ovewrite_primitive_nonprimitive():
    trie = MutableChoiceTrie()
    trie[('a')] = 2
    assert trie.asdict() == {'a': {(): 2}}
    trie[('a', 'b')] = 2
    assert trie.asdict() == {'a': {'b': {(): 2}}}
    subtrie = MutableChoiceTrie()
    subtrie[('c')] = 10
    trie.set_subtrie(('a','b'), subtrie)
    assert trie.asdict() == {'a': {'b': {'c': {(): 10}}}}
    trie[('a','x')] = 1
    assert trie.asdict() == {'a': {'b': {'c': {(): 10}}, 'x': {(): 1}}}
    trie[('a')] = 1
    assert trie.get_subtrie(('a')).has_choice()
    assert trie[('a')] == 1

def test_trie_interactive_session_1():
    trie = MutableChoiceTrie()
    # Create a primitive.
    trie[('a')] = 2
    assert trie.asdict() == {'a': {(): 2}}
    # Create a compound.
    trie[('b', 'c')] = 3
    assert trie.asdict() == {'a': {(): 2}, 'b': {'c': {(): 3}}}
    subtrie = trie.get_subtrie(('b'))
    assert not subtrie.has_choice()
    # Test difference between get_choice and __getitem__
    assert not trie.has_choice()
    with pytest.raises(MutableChoiceTrieError):
        trie.get_choice()
    assert isinstance(trie.get_subtrie(('b')), ChoiceTrie)
    subtrie = trie.get_subtrie(('b', 'c'))
    assert subtrie.has_choice()
    assert subtrie[()] == 3
    assert subtrie.get_choice() == 3
    # Extend a compound.
    trie[('b', 'd')] = 14
    assert trie.asdict() == {'a': {(): 2}, 'b': {'c': {(): 3}, 'd': {(): 14}}}
    with pytest.raises(MutableChoiceTrieError):
        trie[('b')]
    assert trie[('b', 'd')] == 14
    assert trie.get_subtrie(('b')).asdict() == {'c': {(): 3}, 'd': {(): 14}}
    with pytest.raises(MutableChoiceTrieError):
        trie.get_subtrie(('d'))
    subtrie = trie.get_subtrie(('b', 'd'))
    assert subtrie.has_choice()
    assert subtrie[()] == 14
    # Overwrite a primitive with a compound.
    trie[('a', 'c')] = 5
    subtrie = trie.get_subtrie(('a'))
    assert subtrie is trie.get_subtrie(('a'))
    assert not subtrie.has_choice()
    assert subtrie[('c')] == 5
    # Confirm values.
    assert trie[('a', 'c')] == 5
    assert trie[('b', 'c')] == 3
    assert trie[('b', 'd')] == 14
    # Overwrite a compound.
    trie[('b', 'c')] = 13
    assert trie[('b', 'c')] == 13
    # Add a few more address.
    trie[(1)] = 5
    trie.set_subtrie(('a', 'f'), MutableChoiceTrie())
    # Test iteration.
    assert trie.asdict() == {
        'a': {'c': {(): 5}, 'f': {}},
        'b': {'c': {(): 13}, 'd': {(): 14}},
        1: {(): 5}
    }
    expected = [('a',), ('b',), (1,)]
    for k, v in trie.subtries():
        assert k in expected
        expected = [e for e in expected if e != k]
    assert not expected
    expected = [('a','c'), ('b','c'), ('b','d'), (1,)]
    for k, v in trie.choices():
        assert k in expected
        expected = [e for e in expected if e != k]
    assert not expected

def test_trie_primitive_ChoiceTrie_vs_primitive_dict():
    # Writing a dict versus ChoiceTrie as primitive choice.
    # = Write a dict.
    trie_value = {'a' : {(): 1.123}}
    trie = MutableChoiceTrie()
    trie[('b')] =  trie_value
    assert trie.get_subtrie(('b')).has_choice()
    assert trie[('b')] == trie_value
    d1 = trie.asdict()
    assert d1 == {'b': {(): trie_value}}
    # = Write a MutableChoiceTrie.
    trie_value = MutableChoiceTrie()
    trie_value[('a')] = 1.123
    trie = MutableChoiceTrie()
    trie[('b')] = trie_value
    assert trie.get_subtrie(('b')).has_choice()
    assert trie[('b')] == trie_value
    assert trie.get_subtrie(('b'))[()] == trie_value
    d2 = trie.asdict()
    assert d2 == {'b': {(): trie_value}}
    # = Confirm they are not equal
    assert d1 != d2

def test_subtrie_nested():
    trie = MutableChoiceTrie()
    trie[('a')] = 1.123
    trie[('b', 'c')] = 10
    trie[('b', 'e', 1)] = 11
    assert trie.get_subtrie(('b', 'e'))[(1)] == 11
    assert trie.asdict() == {
        'a':
            {(): 1.123},
        'b': {
            'c': {(): 10},
            'e':
                {1: {(): 11}}}
        }
    assert set(trie.choices()) == {
        (('a',), 1.123),
        (('b', 'c'), 10),
        (('b', 'e', 1), 11)
    }
    assert MutableChoiceTrie.copy(trie) == trie

def test_trie_primitive_ChoiceTrie_vs_subtrie():
    # Setting a subtrie versus setting a primitive using index.
    # = setting a primitive that happens to be a choice trie
    trie = MutableChoiceTrie()
    trie[('a')] = MutableChoiceTrie()
    assert trie.asdict() == {'a': {(): MutableChoiceTrie()}}
    assert dict(trie.choices()) == {('a',): trie[('a')]}
    assert dict(trie.subtries()) == {('a',): trie.get_subtrie(('a'))}
    # = setting a subtrie proper
    trie.set_subtrie(('a'), MutableChoiceTrie())
    assert trie.trie == {'a': MutableChoiceTrie()}
    assert trie.asdict() == {'a': {}}
    assert dict(trie.choices()) == {}

def test_flatten_non_empty_subtrie_with_no_choices():
    # Flatten a non-empty subtrie with no choices.
    trie = MutableChoiceTrie()
    trie.set_subtrie(('a'), MutableChoiceTrie())
    assert list(trie.choices()) == []

def test_subtrie_mutation_propagates():
    trie = MutableChoiceTrie()
    trie.set_subtrie(('a'), MutableChoiceTrie())
    subtrie = trie.get_subtrie(('a'))
    subtrie[('b')] = 1.123
    assert trie[('a', 'b')] == 1.123

def test_subtrie_circular_basic():
    trie = MutableChoiceTrie()
    trie[()] = trie
    assert trie.has_choice()
    assert trie[()] == trie

    # Should really be forbidden.
    # The mutation semantics are confusing.
    # Please do not set a trie to include itself.
    trie = MutableChoiceTrie()
    trie.set_subtrie(('a'), trie)
    assert trie.asdict() == {'a': {'a': {}}}
    assert trie.get_subtrie(('a')) is not trie
    trie.set_subtrie(('b'), MutableChoiceTrie())
    trie[('c')] = 1
    assert trie.asdict() == {'a': {'a': {}}, 'b': {}, 'c': {(): 1}}

def test_subtrie_circular_complex():
    trie = MutableChoiceTrie()
    trie[('a')] = 1
    subtrie = MutableChoiceTrie()
    subtrie[('c')] = 'foo'
    trie.set_subtrie(('a'), subtrie)
    assert trie.get_subtrie(('a')) == subtrie
    trie.set_subtrie(('a', 'b'), subtrie)
    assert trie.get_subtrie(('a', 'b')) == subtrie

def test_update_empty_or_primitive():
    # other is primitive
    other = MutableChoiceTrie()
    other[()] = 2
    trie = MutableChoiceTrie()
    trie[()] = 1
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
    trie[()] = 1
    trie.update(other)
    assert trie[()] == 1

    # other is not empty or primitive
    other = MutableChoiceTrie()
    other[('a')] = 2
    inner = MutableChoiceTrie()
    inner[('c')] = 3
    other.set_subtrie(('b'), inner)
    trie = MutableChoiceTrie()
    trie.update(other)
    assert trie == other
    trie = MutableChoiceTrie()
    trie[()] = 1
    trie.update(other)
    assert trie == other
    trie[('a')] = 5
    assert other[('a')] == 2

    # self is primitive
    trie = MutableChoiceTrie()
    trie[('a')] = 1
    other = MutableChoiceTrie()
    other[('a', 'b')] = 2
    trie.update(other)
    trie[('a', 'b')] = 7
    assert other[('a', 'b')] == 2

def test_update_nonprimitive():

    def make_original():
        trie = MutableChoiceTrie()
        trie[('a')] = 1
        trie[('b')] = 2
        inner = MutableChoiceTrie()
        inner[('d')] = 3
        trie.set_subtrie(('c'), inner)
        return trie

    # other is primitive
    other = MutableChoiceTrie()
    other[()] = 2
    trie = make_original()
    trie.update(other)
    assert trie == other
    other[()] = 3
    assert trie[()] == 2

    # other is empty
    other = MutableChoiceTrie()
    trie = make_original()
    trie.update(other)
    assert trie == make_original()

    # other is not empty or primitive
    other = MutableChoiceTrie()
    other[('a')] = 4
    inner = MutableChoiceTrie()
    inner[('e', 'f')] = 5
    other.set_subtrie(('c'), inner)
    trie = make_original()
    trie.update(other)
    expected = MutableChoiceTrie()
    expected[('a')] = 4
    expected[('b')] = 2
    expected[('c', 'd')] = 3
    expected[('c', 'e', 'f')] = 5
    assert trie == expected
    other[('c', 'e', 'f')] = 1
    assert trie == expected
