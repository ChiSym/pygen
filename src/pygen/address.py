class Address:

    def __init__(self, elements):
        self.elements = tuple(elements)
    
    def first(self):
        # TODO error if is empty tuple
        return self.elements[0]

    def rest(self):
        # TODO error if is empty tuple
        return self.elements[1:]

    def isempty(self):
        # TODO to bool (<if addr>)
        return self.elements is ()

def addr(*args):
    return Address(args)

