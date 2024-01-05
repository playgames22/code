import numpy as np

def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

x_range = 10000

X = np.array([
    [0, 0, 1],
    [1, 2, 1],
    [1, 0, 1],
    [0, 1, 1]
])

y = np.array([[1, 0, 2, 1]]).T

np.random.seed(1)
   
syn0 = 2 * np.random.random((3, 1)) - 1
print(syn0)
for _ in range(x_range):
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l1_error = y - l1
    l1_delta = l1_error * nonlin(l1, True)
    syn0 += np.dot(l0.T, l1_delta)

print("Actual Output:\n" + str(y))
print("\nOutput After Training:")
print(l1)
