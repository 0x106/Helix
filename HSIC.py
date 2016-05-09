# HSIC

import numpy as np

N = 40
K = 7

H = np.zeros((N,N))
for i in range(N):
	for k in range(N):
		H[i,k] = 0. - (1./N)
	H[i,i] = 1. - (1./N)
S = np.eye(K) * (1./((K*10000)))
S[6,6] = (1./((K*20)))

def rbf(x, y, S):
	z = x-y
	q = np.dot(z.T, np.dot(S,z))
	return np.exp(-q)

def HSIC(X, Y, H, S):
	
	K,L, KH, LH = np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))

	for i in range(N):
		for k in range(N):
			K[i,k] = rbf(X[:,i], X[:,k], S)
			L[i,k] = rbf(Y[:,i], Y[:,k], S)

	KH = np.dot(K,H)
	LH = np.dot(L,H)

	return ((1. / (N*N)) * np.trace(np.dot(KH,LH))) / (np.sqrt((1. / (N*N)) * np.trace(np.dot(LH,LH))))

def __HSIC__(KH, Y, H, S):
	
	N_ = KH.shape[0]

	K,L, LH = np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))

	for i in range(N):
		for k in range(N):
			L[i,k] = rbf(Y[:,i], Y[:,k], S)

	LH = np.dot(L,H)

	return ((1. / (N*N)) * np.trace(np.dot(KH,LH))) / (np.sqrt((1. / (N*N)) * np.trace(np.dot(LH,LH))))

