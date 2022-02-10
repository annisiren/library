##################
# DICTIONARY

# Self create
{x:x**2 for x in range(10)}

d = {'k1':1,'k2':2}
for k in d.keys():
    print(k)

for v in d.values():
    print(v)

for item in d.items():
    print(item)

key_view = d.keys()

d['k3'] = 3

##################
# LIST

list1 = [1,2,3]

list1.append(4)
list1.count(10)

x = [1, 2, 3]
x.append([4, 5])
print(x)

x = [1, 2, 3]
x.extend([4, 5])
print(x)

list1.index(2)

list1.insert(2,'inserted') # Place a letter at the index 2

ele = list1.pop(1)  # pop the second element

list1.remove('inserted')

list2.reverse()

list2.sort()

y = x.upper()

##################
# SET

s = set()

s.add(1)
s.clear()

s = {1,2,3}
sc = s.copy()
s.add(4)

s.difference(sc)

s1.difference_update(s2)

s.discard(2)


s1 = {1,2,3}
s2 = {1,2,4}
s1.intersection(s2)

s1.intersection_update(s2)


s1 = {1,2}
s2 = {1,2,4}
s3 = {5}
s1.isdisjoint(s2)
s1.isdisjoint(s3)

s1.issubset(s2)
s2.issuperset(s1)
s1.issuperset(s2)

s1.symmetric_difference(s2)

s1.union(s2)

s1.update(s2)
