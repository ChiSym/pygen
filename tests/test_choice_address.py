import pytest

from pygen.choice_address import ChoiceAddress
from pygen.choice_address import addr

def test_address_empty():
    x = addr()
    assert x == ChoiceAddress(())
    assert x == addr()
    assert not x
    with pytest.raises(IndexError):
        x.first()
    with pytest.raises(IndexError):
        x.rest()

def test_address_singleton_empty_tuple():
    x = addr(())
    assert x == ChoiceAddress(((),))
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

def addressify(x):
    if isinstance(x, tuple):
        return addr(*x)
    return addr(x)

def test_addressify():
    assert addressify(('1', '2')) == addr('1', '2')
    assert addressify('1') == addr('1')
    assert addressify(('1',)) == addr('1')
    assert addressify(('1', ())) == addr('1', ())
    assert addressify(()) == addr()
