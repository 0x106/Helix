import numpy as np
import model as ARModel
import util
import cv2
import matplotlib.pyplot as plt

class KL_(object):
	def __init__(self,N):
		self.mean = np.zeros(N)
		self.cov = np.zeros((N,N))

	def __str__(self):
		print 'Mean:'
		print self.mean
		print '==================================================='
		print 'Cov:'
		print self.cov
		print '==================================================='
		print 'Inv:'
		print self.inv
		print '==================================================='
		print 'Det:'
		print self.det
		print '==================================================='

	def set(self):
		self.inv = np.linalg.inv(self.cov)
		self.det = np.linalg.det(self.cov)

def rbf(x,y,cov):
	return np.exp(-(   np.dot((x-y).T, np.dot(cov,(x-y)))  ))

def get_mean_cov(data):

	__mean = np.mean(data, 1)

	for i in range(data.shape[1]):
		data[:,i] -= __mean

	__cov = np.dot(data, data.T) / (data.shape[1] - 1)

	for i in range(data.shape[1]):
		data[:,i] += __mean

	# project into inner product space
	if True:

		N = data.shape[1]
		K = np.zeros((N,N))
		for i in range(N):
			for k in range(N):
				K[i,k] = rbf(data[:,i], data[:,k], __cov)
			# K[i,i] += np.random.rand() * 0.1 - 0.05
			# K[i,:] = np.exp(-np.sum((data[:,i] - data.T)*((np.dot(__cov, (data[:,i] - data.T).T).T).T).T, axis=1))

		mean = np.mean(K, 1)
	
		for i in range(K.shape[1]):
			K[:,i] -= mean

		cov = np.dot(K, K.T) / (K.shape[1] - 1)

		try:
			inv = np.linalg.inv(cov)
		except:
			print 'error here'

		for i in range(K.shape[1]):
			K[:,i] += mean

		# print mean.shape, cov.shape

		return mean, cov



	return __mean, __cov

def get_data(frames, points, dims):
	
	idx = 0

	data = np.zeros((len(points), dims))

	points = points.astype(int)

	data[:,:3]   = frames[0][points[:,1], points[:,0], :]
	data[:,3:6]  = frames[1][points[:,1], points[:,0], :]
	data[:,  6]  = frames[2][points[:,1], points[:,0]]
	data[:, 7:]  = frames[3][points[:,1], points[:,0], :]

	return data.T

def PSO(reference, model, frames, states, dims):

	M = 100      			# number of particles
	K = len(states[0]) 	    # num parameters to define functions

	c1, c2, omega       = 1.49618, 1.49618, 0.7298
	p, v = np.zeros((M, K+1)), np.zeros((M, K))
	b, g = np.zeros((M, K+1)), np.zeros(K+1)

	MAX_ITER = 10

	results = []

	p[0,:K] = states[1][0]
	p[0,K] = KL(reference, model, frames, [states[0], p[0,:K]], dims)

	for i in range(1, M):	
		
		p[i,:K] = states[1] + ((np.random.rand(K) * 0.2) - 0.1)

		p[i,0] *= 100.
		p[i,1] *= 100.

		v[i,:]  = ((np.random.rand(K) * 0.1) - 0.05)

		v[i,0] *= 10.
		v[i,1] *= 10.

		p[i,K] = KL(reference, model, frames, [states[0], p[i,:K]], dims)
		b[i] = np.copy(p[i])

 #    # find the global opt
 	max, idx, = np.copy(p[0,K]), 0
 	for i in range(1, M):
 		if p[i,K] > max:
 			max, idx = np.copy(p[i,K]), i
 	g = np.copy(p[idx,:])

	# ## ==== RUN PSO ==== ##
	for iter in range(MAX_ITER):

		print '-->', iter, g[K]

		for i in range(M):
			# p[i,K]      = compute_energy(p[i,:K], initial_model, params, P_KH, N_KH, HS, display=False)
			p[i,K] = KL(reference, model, frames, [states[0], p[i,:K]], dims)
			if p[i,K] > b[i,K]:
				b[i,:] = np.copy(p[i,:])

		max, idx = np.copy(g[K]), -1
		for i in range(0, M):
			if p[i,K] > max:
				max, idx = np.copy(p[i,K]), i

		if idx > -1:
			g = np.copy(p[idx,:])

		# compute_energy(g[:K], initial_model, params, P_KH, N_KH, HS, display=True)
		KL(reference, model, frames, [states[0], g[:K]], dims, draw=True)

		for i in range(M):
			r1, r2 = np.random.rand(1)[0], np.random.rand(1)[0]
			v[i] = (omega * np.copy(v[i])) + (c1 * r1 * (np.copy(b[i,:K]) - np.copy(p[i,:K]))) + (c2 * r2 * (np.copy(g[:K]) - np.copy(p[i,:K])))
			p[i,:K] += np.copy(v[i])

	# compute_energy(g[:K], initial_model, params, P_KH, N_KH, HS, display=True)

	states[1] = np.copy(g[:K])

	return states

def KL(reference, model, frames, state, dims, draw=False):

	mean, cov = np.zeros(dims), np.zeros((dims,dims))

	query_state = state[1]

	model = ARModel.wiggle(model, query_state)
	query = model.get_points(all_points=True)

	if draw:
		display = model.draw(np.copy(frames[1][0]))

	model = ARModel.wiggle(model, query_state, inverse=True)

	query_data = get_data(frames[1], query, dims)

	mean, cov = get_mean_cov(query_data)

	if draw:
		cv2.imshow("KL PSO", display)
		cv2.imshow("KL 1", frames[1][1])
		cv2.imshow("KL 2", frames[1][2])
		cv2.waitKey(1)

	result = 0.

	a = np.trace(np.dot(reference.inv, cov))
	b = np.dot((reference.mean - mean), np.dot(reference.inv, (reference.mean - mean).T))
	c = dims
	d = np.log(reference.det / np.linalg.det(cov))

	# print a, b, c, d, -0.5 * (a + b - c + d)
	return -0.5 * (a + b - c + d)
	# return a, b, c, d, -0.5 * (a + b - c + d)

# global params
params = util.parameters()
image, hsv, dt, flow = params.get_frames()

# model params
model = ARModel.articulated_model(image.shape)
state = [[0. for i in range(15)], [0. for i in range(15)]]#np.random.rand(dims) * 0.25 - 0.125]
frames = [params.get_frames(), params.get_frames(6)]

# the reference example for tracking
ref = KL_(9)

# initialise the reference model
model = ARModel.wiggle(model, state[0])
reference = model.get_points(all_points=True)
model = ARModel.wiggle(model, state[0], inverse=True)

# set the reference params based on the data
ref_data = get_data(frames[0], reference, 9)
ref.mean, ref.cov = get_mean_cov(ref_data)
ref.set()

state = PSO(ref, model, frames, state, 9)

# # ===== OPTIMISE ===== #

# for i in range(4):
# 	state = PSO(ref, model, frames, state, 9)
# 	frames[1] = params.get_frames(1)

# ----- sweep test ----- #
if(True):
	state[1][2] -= 0.01 * 200
	results = np.zeros((1,400))

	for i in range(400):
		# frames[1] = params.get_frames(1)
		results[:,i] = KL(ref, model, frames, state, 9, draw=True)
		state[1][2] += 0.01

	for i in range(1):
		plt.plot(results[i,:])
	plt.show()









