import numpy as np

'''
Erstellen
'''
# from existing dat

np.array([1, 2, 3], dtype=complex)
np.array([[1, 2, 3], [4, 5, 6]])
np.array([[1, 2, 3], [4, 5]])  # Achtung!
np.array([[1, 2, 3], [4, 5, 6]]).reshape(3, 2)

np.asarray([1, 2, 3])
np.asarray([1, 2, 3], dtype=float)
np.asarray((1, 2, 3))
np.asarray([(1, 2, 3), (4, 5)])

# uninitialisiert, Nullen, Einsen
np.empty([3, 2], dtype=int)
np.zeros((2, 3), dtype=float, order='C')
np.ones((2, 2), dtype=int)

# gleichmäßig verteilte Zahlen
np.arange(24)
np.arange(2).dtype == 'int64'
np.arange(2).dtype == 'int32'
np.linspace(10, 20, 5)

# with repetition
np.array([1, 2, 3] * 3)
np.repeat([1, 2, 3], 3)

# combining arrays
p = np.ones([2, 3], int)
np.vstack([p, 2 * p])
np.hstack([p, 2 * p])

# copy existing array
r = np.arange(12).reshape(4, 3)
r.copy()  # important step when modifying slices

'''
Indexing

i:j:k where 
i is the starting index, 
j is the stopping index (excl.), and 
k is the step (default 1)
'''
x = np.arange(10)
x[2:5]
x[1:7:2]

# "Fancy indexing" = indexing using integer arrays or lists.
a = np.arange(6) + 1
indices = [0, 2, 4]
a[indices]

np.take(a, indices)

# Multi-dimensional
A = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
A[:, 1]
A[1, :]

y = np.arange(35).reshape(5, 7)
y
y[1:5:2, ::3]
y[np.array([0, 2, 4]), np.array([0, 1, 2])]
y[[0, 2, 4], [0, 1, 2]]
y[[0, 2, 4], np.array([0, 1, 2])]

# Combining index arrays with slices
y[np.array([0, 2, 4]), 1:3]

'''
Select values from a matrix by an index list. 
Inspiriert durch das GAP und den GA dazu, 
bei dem eine Lösung durch ein Gen repräsentiert wird.
Um die Kosten zu bestimmen, muss man die ensprechenden Einträge
aufsummieren: summe_j c_{g_j, j}.
'''
c = np.array([1, 3, 5, 2, 4, 6]).reshape(2, 3)
c
g = np.array([0, 0, 1])
costs = 0
for j in range(3):
    costs += c[g[j]][j]
costs
c[g, np.arange(3)]
c[g, np.arange(3)].sum()
c[g, :]

# capacity used per agent
# ?


a = np.arange(36).reshape(6, 6)
a
a[:, [2, 3]]
a[[2, 3], :]
a[[2, 3], [2, 3]]  # ! Achtung: <> a[2:4, 2:4]
a[:, [2, 3]][[2, 3], :]

''' Zufälliges Element '''
np.random.choice([1, 2, 3])
np.random.choice(5, 3)  # random sample from np.arange(5) of size 3

'''
Comparison
'''
np.array([1, 2]) == np.array([3, 2])
np.array_equal([1, 2], [1, 2])

'''
toString
'''
np.array([1, 2]).tostring()
np.array_str(np.array([1, 2]))
np.array2string(np.array([1, 2]))
np.array2string(np.array([1, 2]), separator=',')

''' Iterating Over Arrays '''
for val in np.arange(12):
    print(val)

for row in np.arange(6).reshape(2, 3):
    print(row)

a = np.arange(3)
for i in range(len(a)):
    print(a[i])

for i, row in enumerate(np.arange(6).reshape(2, 3)):
    print('row', i, 'is', row)

for i, j in zip(np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3) * -1):
    print(i, '+', j, '=', i + j)

'''
Statistical Functions
'''
print("\n\nStats")
a = np.array([[3, 7, 5], [8, 4, 3], [2, 4, 9]])
print(a)
print(np.amin(a))
print(np.amin(a, 0))
print(np.amin(a, 1))

# Range
print(np.ptp(a))
print(np.ptp(a, 0))
print(np.ptp(a, 1))

print(np.percentile(a, 50))
print(np.mean(a))

# weighted avg
a = np.array([1, 2, 3, 4])
print(np.average(a, weights=(0.25, 0.25, 0.25, 0.25)))

print(np.sqrt(np.mean(abs(a - a.mean()) ** 2)))
print(np.std([1, 2, 3, 4]))
print(np.var([1, 2, 3, 4]))

# find index
a = np.array([1, 2, 3, 3, 1])
result = np.where(a == np.amax(a))  # returns tuple of arrays (one for each axis)
result
result[0]

'''
Linear Algebra
'''

A = [[1, 2, 3],
     [4, 5, 6]]
B = [[0, 1, 2],
     [2, 3, 3]]
C = np.add(A, B)
C

x = np.asarray([3, 2])
y = np.asarray([1, 3])
x
y
MAE = np.sum(np.abs(np.subtract(x, y)))
x <= y
assert (x <= y).all()
assert (x <= y).any()
assert np.logical_and(-2 <= x, x <= 4).all()

x.dot(y)
x * y

# Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)
A.dot(x) == b

# Ax = b (Alternative)
inv_A = np.linalg.inv(A)
x = inv_A.dot(b)

# eigene Funktion mit Fallunterscheidung
x = np.array([1, 1.5, 2, 2.5])
np.where(x % 1 == 0, -x, x)

'''
Praktische Funktionen
'''
a = np.arange(10)
a
np.clip(a, 2, 6)

''' 
Strings
'''
c = np.array(['a1b2', '1b2a', 'b2a1', '2a1b'])
c
np.char.capitalize(c)
