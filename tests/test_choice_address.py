import pytest

from pygen.choice_address import addr
from pygen.choice_address import addressify

def test_address_empty():
    x = addr()
    assert x == addr()
    assert x.keys == ()
    assert not x
    with pytest.raises(RuntimeError):
        x.first()
    with pytest.raises(RuntimeError):
        x.rest()

def test_address_singleton_empty_tuple():
    x = addr(())
    assert x == addr(())
    assert x.keys == ((),)
    assert x
    assert x.first() == ()
    assert x.rest() == addr()

def test_address_singleton_non_empty_tuple():
    x = addr(('a',))
    assert x
    assert x.keys == (('a',),)
    assert x.first() == ('a',)
    assert x.rest() == addr()

def test_address_pair():
    x = addr('a', 'b')
    assert x
    assert x.keys == ('a', 'b')
    assert x.first() == 'a'
    assert x.rest() == addr('b')
    assert x.rest().rest() == addr()

def test_addressify():
    assert addressify(()) == addr()

    assert addressify(1) == addr(1)
    assert addressify(1,) == addr(1)
    assert addressify((1,)) == addr(1)

    assert addressify((1, 2)) == addr(1, 2)
    assert addressify((1,2),) == addr(1, 2)

    assert addressify(((1,2),3)) == addr((1, 2), 3)

    assert addressify((1, ())) == addr(1, ())

    assert addressify(addr(1,2)) == addr(1,2)
    assert addressify(addr(1,2),) == addr(1,2)
    assert addressify((addr(1,2),)) == addr(addr(1,2))

    with pytest.raises(TypeError):
        addressify(1,2)

class DummyTrie:
    def __getitem__(self, key):
        return [key, addressify(key)]

def test_addressify_getitem():
    trie = DummyTrie()

    assert trie[1] == [1, addr(1)]
    assert trie[1,] == [(1,), addr(1)]

    assert trie[1,2] == [(1,2), addr(1,2)]
    assert trie[1,2,] == [(1,2) , addr(1,2)]

    assert trie[(1,2)] == [(1,2), addr(1,2)]
    assert trie[(1,2),] == [((1,2),), addr((1,2))]
    assert trie[(1,2),3] == [((1,2),3), addr((1,2),3)]

    assert trie[addr(1)] == [addr(1), addr(1)]
    assert trie[addr(1),] == [(addr(1),), addr(addr(1))]
    assert trie[addr(1),2] == [(addr(1),2), addr(addr(1),2)]
