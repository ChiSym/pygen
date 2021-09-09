class ChoiceAddress:

    def __init__(self, keys):
        self.keys = tuple(keys)

    def first(self):
        if not self:
            raise IndexError
        return self.keys[0]

    def rest(self):
        if not self:
            raise IndexError
        return ChoiceAddress(self.keys[1:])

    def __bool__(self):
        return bool(self.keys)

    def __eq__(self, x):
        return isinstance(x, type(self)) and self.keys == x.keys

    def __repr__(self):
        return 'ChoiceAddress(%s)' % (self.keys,)

    def __str__(self):
        return str(self.keys)

    def __hash__(self):
        return hash(self.keys)

    def __add__(self, x):
        if not isinstance(x, type(self)):
            return NotImplemented
        return ChoiceAddress(self.keys + x.keys)

def addr(*args):
    return ChoiceAddress(args)

import pytest

x = addr('a', 'b')
assert x
assert x.keys == ('a', 'b')
assert x.first() == 'a'
assert x.rest() == addr('b')
assert x.rest().rest() == addr()

x = addr(('a',))
assert x
assert x.keys == (('a',),)
assert x.first() == ('a',)
assert x.rest() == addr()

x = addr()
assert x == ChoiceAddress(())
assert x == addr()
assert not x
with pytest.raises(IndexError):
    x.first()
with pytest.raises(IndexError):
    x.rest()

x = addr(())
assert x == ChoiceAddress(((),))
assert x
assert x.first() == ()
assert x.rest() == addr()

def addressify(x):
    if isinstance(x, tuple):
        return addr(*x)
    return addr(x)

assert addressify(('1', '2')) == addr('1', '2')
assert addressify('1') == addr('1')
assert addressify(('1',)) == addr('1')
assert addressify(('1', ())) == addr('1', ())
assert addressify(()) == addr()
