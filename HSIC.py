# HSIC

import numpy as np

# N = 40
# K = 7

# H = np.zeros((N,N))
# for i in range(N):
# 	for k in range(N):
# 		H[i,k] = 0. - (1./N)
# 	H[i,i] = 1. - (1./N)
# S = np.eye(K) * (1./((K*10000)))
# S[6,6] = (1./((K*20)))

# def rbf(x, y, S):
# 	z = x-y
# 	q = np.dot(z.T, np.dot(S,z))
# 	return np.exp(-q)

# def HSIC(X, Y, H, S):
	
# 	K,L, KH, LH = np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))

# 	for i in range(N):
# 		for k in range(N):
# 			K[i,k] = rbf(X[:,i], X[:,k], S)
# 			L[i,k] = rbf(Y[:,i], Y[:,k], S)

# 	KH = np.dot(K,H)
# 	LH = np.dot(L,H)

# 	return ((1. / (N*N)) * np.trace(np.dot(KH,LH))) / (np.sqrt((1. / (N*N)) * np.trace(np.dot(LH,LH))))

# def __HSIC__(KH, Y):
	
# 	N_ = KH.shape[0]

# 	K,L, LH = np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))

# 	for i in range(N):
# 		for k in range(N):
# 			L[i,k] = rbf(Y[:,i], Y[:,k], S)

# 	LH = np.dot(L,H)

# 	return ((1. / (N*N)) * np.trace(np.dot(KH,LH))) / (np.sqrt((1. / (N*N)) * np.trace(np.dot(LH,LH))))


class HilbertSchmidt(object):

	N = 10
	K = 7
	H = np.zeros((N,N))
	S = np.eye(K) * (1./((K*10000)))
	S[6,6] = (1./((K*20)))

	def __init__(self, N, _K=7):
		self.N = N
		self.H = np.zeros((self.N,self.N))
		for i in range(self.N):
			for k in range(self.N):
				self.H[i,k] = 0. - (1./self.N)
			self.H[i,i] = 1. - (1./self.N)

		if _K != 7:
			self.S = np.eye(_K) * (1./((_K*20000.)))
		else:
			self.S = np.eye(_K) * (1./((_K*10000)))
			self.S[6,6] = (1./((K*20)))

	def __HSIC__(self, KH, Y):
	
		if np.sum(Y) == 0:
			return 0.

		L, LH = np.zeros((self.N,self.N)), np.zeros((self.N,self.N))

		for i in range(self.N):
			for k in range(self.N):
				L[i,k] = self.rbf(Y[:,i], Y[:,k])

		LH = np.dot(L,self.H)

		return ((1. / (self.N*self.N)) * np.trace(np.dot(KH,LH))) / (np.sqrt((1. / (self.N*self.N)) * np.trace(np.dot(LH,LH))))

	def rbf(self, x, y):
		z = x-y
		q = np.dot(z.T, np.dot(self.S,z))
		return np.exp(-q)

	def get_H(self):
		return np.copy(self.H)

	def get_N(self):
		return self.N

	def get_K(self):
		return self.K







