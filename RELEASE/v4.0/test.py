import numpy as np

N = 100000
A = np.random.rand(N)
B = np.random.rand(N)

print np.dot(A.T, B)
