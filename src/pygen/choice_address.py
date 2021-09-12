class ChoiceAddress:

    def __init__(self, *keys):
        self.keys = tuple(keys)

    def first(self):
        if not self:
            raise RuntimeError('Empty ChoiceAddress has no first() element.')
        return self.keys[0]

    def rest(self):
        if not self:
            raise RuntimeError('Empty ChoiceAddress has no rest() address.')
        return ChoiceAddress(*self.keys[1:])

    def __bool__(self):
        return bool(self.keys)

    def __eq__(self, x):
        if isinstance(x, type(self)):
            return self.keys == x.keys
        return False

    def __repr__(self):
        x = ', '.join(repr(x) for x in self.keys)
        return 'addr(%s)' % (x,)

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.keys)

addr = ChoiceAddress

def addressify(x):
    if isinstance(x, ChoiceAddress):
        return x
    if isinstance(x, tuple):
        return addr(*x)
    return addr(x)
