import numpy as np

def max_n(a, n):
    return np.partition(a.flatten(), -n)[-n]

a = np.array([1, 5, 2, 4, 11, 13, -5, 2, 13])

print(max_n(a, 1))
print(max_n(a, 2))
print(max_n(a, 3))
print(max_n(a, 4))
print(max_n(a, 5))

q_table = np.zeros(shape=((2, 3) + (4,)))
print(q_table)
