from pygen.choice_trie import MutableChoiceTrie
from pygen.choice_address import addr

t = MutableChoiceTrie()

print(t)

t[addr('x')] = 1
print(t)

t[addr('x', 'y')] = 1
print(t)

t[addr('x')] = 1
print(t)

t[addr('x', 'y')] = 2
print(t)

#t[] = 3 # < not supported.. instead use t.set_value(3)?

t[addr('x', 'y')] = 2
t[addr('x', 'z')] = 3
t[addr('x', 'z', 'w')] = 4
args = ('x', 'z', 'w')
t[addr(*args)] = 123
print(t)

print("subtries")
kvs = [kv for kv in t.subtries()]
assert len(kvs) == 1
print(kvs)

print("asdict")
print(t.asdict())
