
def unpack_tuple_change(value, change):
    if change == Same():
        return tuple(Same() for _ in value)
    elif change == Unknown():
        return tuple(Same() for _ in value)
    elif isinstance(change, tuple):
        return change
    else:
        raise NotImplementedError(f'invalid change value for tuple: {change}')

# TODO what about changing the length of a tuple?

# TODO list change?

class Change:
    pass


class Same(Change):

    def __eq__(self, other):
        return isinstance(other, Same)


class Unknown(Change):

    def __eq__(self, other):
        return isinstance(other, Unknown)