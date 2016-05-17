import numpy as np
import matplotlib.pyplot as plt

N = 100

X1 = np.linspace(0,N,num=N)
X2 = np.linspace(0,N,num=N)

Y1 = np.zeros(N)
Y2 = np.zeros(N)
Y4 = np.zeros(N)

Y1[:N/2] = X1[:N/2]
Y1[N/2:] = -X1[N/2:] + 2*X1[N/2]

Y2[:90] = Y1[10:] / 4.
Y4[20:] = /
Y1[10:-10] / 4.

Y3 = Y1 - Y2 - Y4

plt.plot(Y1)
plt.plot(Y2)
plt.plot(Y3)
plt.plot(Y4)
plt.show()
