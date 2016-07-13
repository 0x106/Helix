import numpy as np

A = np.random.rand(1000)
B = np.random.rand(1000)

M = 100000000

for i in range(M):
    C = np.dot(A, B)

