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
        if isinstance(x, type(self)):
            return self.keys == x.keys
        return False

    def __repr__(self):
        return 'ChoiceAddress(%s)' % (self.keys,)

    def __str__(self):
        return str(self.keys)

    def __hash__(self):
        return hash(self.keys)

    def __add__(self, x):
        if isinstance(x, type(self)):
            return ChoiceAddress(self.keys + x.keys)
        return NotImplemented

def addr(*args):
    return ChoiceAddress(args)
