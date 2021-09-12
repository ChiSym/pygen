from pygen.choice_trie import MutableChoiceTrie, choice_trie
from pygen.choice_address import addr

trie = MutableChoiceTrie()

t = trie.flat_view()

t = choice_trie()
print(t.hierarchical_view)

t['x'] = 1
print(t.hierarchical_view)

t['x', 'y'] = 1
print(t.hierarchical_view)

t['x'] = 1
print(t.hierarchical_view)

t['x', 'y'] = 2
print(t.hierarchical_view)

#t[] = 3 # < not supported.. instead use t.set_value(3)?
#print(t.hierarchical_view)

t['x', 'y'] = 2
t['x', 'z'] = 3
t[('x', 'z', 'w')] = 4
args = ('x', 'z', 'w')
t['x', 'z', 'w'] = 4
t[(*args,)] = 123

print("choice trie")
print(t.hierarchical_view)

print("flat iterator")
kvs = [kv for kv in t]
assert len(kvs) == 2
print(kvs)

#print(t.hierarchical_view)

#kvs = [kv for kv in t.hierarchical_view]
print(kvs)

print(t.hierarchical_view[addr('x')])
print(t.hierarchical_view[addr('x', 'y')])
print(t.hierarchical_view[addr('x', 'z')])
#print(t.hierarchical_view[addr('u', 'z')]) # errors as it should
print(t.hierarchical_view.get(addr('u', 'z'), strict=False))
