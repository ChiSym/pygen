from pygen.choice_trie import MutableChoiceTrie, choice_trie
from pygen.choice_address import addr

trie = MutableChoiceTrie()

t = trie.flat_view()

t = choice_trie()
print(t.choice_trie)

t['x'] = 1
print(t.choice_trie)

t['x', 'y'] = 1
print(t.choice_trie)

t['x'] = 1
print(t.choice_trie)

t['x', 'y'] = 2
print(t.choice_trie)

#t[] = 3 # < not supported.. instead use t.set_value(3)?
#print(t.choice_trie)

t['x', 'y'] = 2
t['x', 'z'] = 3
t[('x', 'z', 'w')] = 4
args = ('x', 'z', 'w')
t['x', 'z', 'w'] = 4
t[(*args,)] = 123
print(t.choice_trie)

kvs = [kv for kv in t]
assert len(kvs) == 2
print(kvs)

print(t.choice_trie)

kvs = [kv for kv in t.choice_trie]
print(kvs)

print(t.choice_trie[addr('x')])
print(t.choice_trie[addr('x', 'y')])
print(t.choice_trie[addr('x', 'z')])
#print(t.choice_trie[addr('u', 'z')]) # errors as it should
print(t.choice_trie.get(addr('u', 'z'), strict=False))
