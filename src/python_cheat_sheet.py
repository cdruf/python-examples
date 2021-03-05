import doctest
import unittest
from functools import reduce

# %%

"""
# Variables, Numbers
"""

a = 1
b = 2
a, b = b, a
print(a, b)
print(a, b, sep=', ')

print("Spam " * 4)

print(type(1))
print(type(1.1))

print(min(1, 2, 3))
print(max(1, 2, 3))

print(abs(-32))

print(int('807') + 1)  # cast to int

# %%

"""
# Strings
"""
# Format
print('%d -> %d' % (1, 2))
print('{} -> {}'.format(1, 2))

# Fill
str(3).zfill(4)
'1'.ljust(4, '0')
'1'.rjust(4, '0')

# %%

"""
# Functions
"""

help(round)


# Test via doc-string (doctest)
def least_difference(a, b, c):
    """
    A least difference function.
    
    Define a test with the doctest module:
    >>> least_difference(1, 3, 6)
    2
    """
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    return min(diff1, diff2, diff3)


doctest.testmod()

print(least_difference(1, 3, 6))

help(least_difference)


# Test via subclassing TestCase
# Tests are defined with methods whose names start with the letters test. 
# This naming convention informs the test runner about which methods represent tests.
class TestFunction(unittest.TestCase):

    def test_1(self):
        self.assertEqual(least_difference(1, 3, 6), 2)


unittest.main()

# %%

"""
# Booleans und Bedingungen
"""

# True, False

# Vergleichsoperatoren wie Java

# elif

# conversion
print(bool(1))  # all numbers are treated as true, except 0
print(bool(0))
print(bool("asf"))  # all strings are treated as true, except the empty string ""
print(bool(""))

print(sum([True, False, True]))  # 2

# ternary
print('failed' if 1 < 50 else 'passed')

# %%

"""
# Lists
"""

sum([1, 2])

planets = ['Mercury', 'Uranus', 'Nept']
planets[-1] = 'Neptune'
planets[1:]
len(planets)

planets.append('Pluto')
planets.extend(['Mars', 'Erde'])
planets += ['Jupiter', 'Venus']
planets.pop()  # letztes!
planets.pop(0)  # erstes
planets
planets.index('Neptune')

'Uranus' in planets

[1] * 3  # repeat

# zip, unzip
list1 = [5, 3, 1, 4, 2]
list2 = ['five', 'three', 'one', 'four', 'two']

list(zip(list1, list2))
l1, l2 = zip(*zip(list1, list2))

# print
'| '.join(['%s, %s' % (j, i) for i, j in zip(list1, list2)])

# sort 
sorted(planets)
sorted(planets, key=lambda x: x[-1], reverse=True)
print(sorted(zip(list1, list2), key=lambda x: x[0], reverse=True))
print(*sorted(zip(list1, list2), key=lambda x: x[0], reverse=True))
list1, list2 = zip(*sorted(zip(list1, list2), key=lambda x: x[0], reverse=True))
list1
list2


# Map
doubled_list = map(lambda x: x*2, list1)
print(', '.join([str(x) for x in doubled_list]))


# Reduce
summation = reduce(lambda x, y: x + y, list1, 0)
summation

# Any, all
all([0, 1, 2])  # 0 = False
any([0, 1, 2])

# Filter
iterator = filter(lambda x: x != 2, [1, 2, 3])
list(iterator)


# %%

"""
# Tuples --- immutable
"""

a = [1, 2]
a[0], a[1] = a[1], a[0]
a

# %%

"""
# Sets
"""

help(set)

set([1, 1, 2, 3])

# ...

# %%


###
# List comprehensions
###
[n ** 2 for n in range(10)]
[n * 50 for n in range(1, 9)]
[(i, j) for i in range(2) for j in range(3)]
[[j for j in range(9, 11)] for i in range(2)]

ns = [n * 50 for n in range(1, 9)]
results = {}
for n in ns: results[n] = 'asdf'
results

[n * (n + 1) for n in range(5)]

[(i, j) for i in ['a', 'b'] for j in [1, 2]]  # sequence

import gurobipy as grb

arcs = grb.tuplelist([('a', 'b'), ('b', 'c'), ('a', 'c')])
[arc for arc in arcs.select('a', '*')]
[arc for arc in arcs if arc[0] == 'a']  # less efficient

# %%

"""
# Dictionaries
"""

# Create
w = {'a': 1, 'b': 2}
dict([('a', 1), ('b', 2)])  # from list of tuples
dict(zip(['a', 'b', 'c'], [1, 2, 3]))  # from key and value lists
dict.fromkeys(['a', 'b'], 0)

# dynamic views
print(w.items())
print(w.keys())
print(w.values())
keys = w.keys()
w['c'] = 3
print(keys)  # including c!

# compare dictionaries
a = {'a': 1, 'b': 2}
b = {'a': 1, 'b': 2}
a == b  # True
b['b'] = 3
a == b  # False
b['b'] = 2;
b['c'] = 3
a == b  # False

# filter by condition on key
dict(filter(lambda x: x[0] == 'a', w.items()))
sum(v for k, v in filter(lambda x: x[0] != 'x', w.items()))

# default values
from collections import defaultdict

d = defaultdict(lambda: -1)
print(d[0])
d = defaultdict(list)
print(d[0])
d[1].append(1)
print(d[1])

d = {}
d.setdefault(0, -1)
d[0]
print(d)

# Nested dictionary and items
d = {'a': {'x': 1}, 'b': {'y': 2, 'z': 3}}
lst = ['{}, {}, {}'.format(k1, k2, v2) for k1, v1 in d.items()
       for k2, v2 in v1.items()]
print(lst)


# %%

###
# System calls
###
import subprocess

subprocess.run(["ls", "-l"])

###
# datetime
###

# from timestamp to dt
from datetime import datetime

timestamp = 1545730073
dt_object = datetime.fromtimestamp(timestamp)
print("dt_object =", dt_object)
print("type(dt_object) =", type(dt_object))

# from dt to timestamp
now = datetime.now()
timestamp = datetime.timestamp(now)
print("timestamp =", timestamp)

##
# Map
###
help(map)
store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2)
type(cheapest)
for item in cheapest:
    print(item)


# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid) -> float:
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)


# %%

"""
# Loops
"""

for idx, val in enumerate([1, 4, 5]):
    print(idx, val)


# %%
# Classes

class X:
    def __init__(self):
        self.x = 1
        self.d = {'a': 1, 'b': 2}

    def __str__(self):
        return str(self.__dict__)


class A:
    def __init__(self):
        self.x = X()
        x = self.x
        x.x += 1
        print(x.x)
        self.y = 1
        y = self.y
        y += 1
        print(y)

        self.lst = [X(), X()]

    def __str__(self):
        return str([str(i) for i in self.lst])


# call-by-value
a = A()
print(a.x.x)
print(a.y)

# print
print(a)

[str(i) for i in a.lst]
# %%




# TODO: put in the correct spot
april_1st = datetime(2021, 4, 1)
print(f"April the 1st is a {april_1st.strftime('%A')}")
print(f"April the 0st in calendar week {april_1st.isocalendar()[1]}")
