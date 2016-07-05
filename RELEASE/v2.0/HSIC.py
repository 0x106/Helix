import numpy as np

class HilbertSchmidt(object):

	def __init__(self, _N, _K):
		self.N = _N
		self.K = _K
		self.H = np.zeros((self.N,self.N))
		for i in range(self.N):
			for k in range(self.N):
				self.H[i,k] = 0. - (1./self.N)
			self.H[i,i] = 1. - (1./self.N)

	def HSIC(self, KH, LH):
		_KH_ = np.ravel(KH)
		_LH_ = np.ravel(LH)

		scale = 1. / self.N
		
		A = scale * np.dot(_KH_, _LH_)
		B = scale * np.dot(_KH_, _KH_)
		C = scale * np.dot(_LH_, _LH_)

		return A / np.sqrt(B * C)

	def get_H(self):
		return np.copy(self.H)

	def get_N(self):
		return self.N

	def get_K(self):
		return self.K

	def get_cov(self):
		return self.cov

	def get_mean_cov(self):
		return self.mean, self.cov

	def set_covariance(self,data, type=0):
		self.mean = np.mean(data, 1)
	
		for i in range(data.shape[1]):
			data[:,i] -= self.mean

		self.cov = np.dot(data, data.T) / (data.shape[1] - 1)

		# try:
		# 	self.cov = np.linalg.inv(cov)
		# except:
		# 	pass

		for i in range(data.shape[1]):
			data[:,i] += self.mean

