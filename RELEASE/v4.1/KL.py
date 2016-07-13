import numpy as np
import model as ARModel
import util
import cv2
import matplotlib.pyplot as plt

def PSO(state, model, frames, grid, sf, w, h, H, N, _KH_2, B2):

	M = 50      # number of particles
	K = len(state[0]) 	      # num parameters to define functions

	c1, c2, omega       = 1.49618, 1.49618, 0.7298
	p, v = np.zeros((M, K+1)), np.zeros((M, K))
	b, g = np.zeros((M, K+1)), np.zeros(K+1)

	MAX_ITER = 10

	results = []

	p[0,:K] = state[0]
	model = ARModel.wiggle(model, p[0,:])
	query = model.get_points(all_points=True)
	image, hsv, edges, dt = create_image(query, w*sf,h*sf, np.copy(frames[0][0]))
	image = cv2.resize(image, (w,h))
	dt = cv2.resize(dt, (w,h))
	model = ARModel.wiggle(model, p[0,:], inverse=True)
	p[0,K] = HSIC([image, dt], grid, H, N, _KH_2, B2)#compute_energy(p[0,:K], initial_model, params, P_KH, N_KH, HS)

	for i in range(1, M):	
		
		p[i,:K] = state[0] + ((np.random.rand(K) * 0.2) - 0.1)
		p[i,0] *= 100.
		p[i,1] *= 100.

		v[i,:]  = ((np.random.rand(K) * 0.1) - 0.05)

		v[i,0] *= 10.
		v[i,1] *= 10.

		model = ARModel.wiggle(model, p[i,:])
		query = model.get_points(all_points=True)
		image, hsv, edges, dt = create_image(query, w*sf,h*sf, np.copy(frames[0][0]))
		image = cv2.resize(image, (w,h))
		dt = cv2.resize(dt, (w,h))
		model = ARModel.wiggle(model, p[i,:], inverse=True)
		p[i,K]   = HSIC([image, dt], grid, H, N, _KH_2, B2)#compute_energy(p[i,:K], initial_model, params, P_KH, N_KH, HS)
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
			model = ARModel.wiggle(model, p[i,:])
			query = model.get_points(all_points=True)
			image, hsv, edges, dt = create_image(query, w*sf,h*sf, np.copy(frames[0][0]))
			image = cv2.resize(image, (w,h))
			dt = cv2.resize(dt, (w,h))
			model = ARModel.wiggle(model, p[i,:], inverse=True)
			p[i,K] = HSIC([image, dt], grid, H, N, _KH_2, B2)#compute_energy(p[i,:K], initial_model, params, P_KH, N_KH, HS, display=False)
			if p[i,K] > b[i,K]:
				b[i,:] = np.copy(p[i,:])

		max, idx = np.copy(g[K]), -1
		for i in range(0, M):
			if p[i,K] > max:
				max, idx = np.copy(p[i,K]), i

		if idx > -1:
			g = np.copy(p[idx,:])

		# this call is just to draw the current global opt to the screen. The value is computed / copied earlier
		model = ARModel.wiggle(model, g)
		query = model.get_points(all_points=True)
		image, hsv, edges, dt = create_image(query, w*sf,h*sf, np.copy(frames[0][0]))
		model = ARModel.wiggle(model, g, inverse=True)
		image = cv2.resize(image, (w,h))
		dt = cv2.resize(dt, (w,h))
		g[K] = HSIC([image, dt], grid, H, N, _KH_2, B2)#compute_energy(g[:K], initial_model, params, P_KH, N_KH, HS, display=True)

		for i in range(M):
			r1, r2 = np.random.rand(1)[0], np.random.rand(1)[0]
			v[i] = (omega * np.copy(v[i])) + (c1 * r1 * (np.copy(b[i,:K]) - np.copy(p[i,:K]))) + (c2 * r2 * (np.copy(g[:K]) - np.copy(p[i,:K])))
			p[i,:K] += np.copy(v[i])

	# HSIC([image, dt], [y,x], H, N, _KH_2, B2)#compute_energy(g[:K], initial_model, params, P_KH, N_KH, HS, display=True)

	state[1] = np.copy(g[:K])

	model = ARModel.wiggle(model, g)
	query = model.get_points(all_points=True)
	image, hsv, edges, dt = create_image(query, w*sf,h*sf, np.copy(frames[0][0]))
	model = ARModel.wiggle(model, g, inverse=True)
	display = add_points(query, np.copy(frames[0][0]))
	display[[y,x][0].astype(int),[y,x][1].astype(int)] = [0,255,0]
	cv2.imshow("PSO", display)
	cv2.waitKey(0)

	return state

def notes():

	print ''' 

1. DONE --> Define a grid of points over each joint and then take those that lie within
	   the defined ellipse.

2. HSIC - only take the k nearest neighbours when constructing K / L etc
			- i.e. so that each point is only compared to a local region.

'''

w, h = 320, 240

cv2.namedWindow("Image")
cv2.moveWindow("Image", 10,10)

cv2.namedWindow("DT")
cv2.moveWindow("DT", 20+w,10)

cv2.namedWindow("Image Data")
cv2.moveWindow("Image Data", 10,50+h)

cv2.namedWindow("DT Data")
cv2.moveWindow("DT Data", 20+w,50+h)

cv2.namedWindow("Display")
cv2.moveWindow("Display", 10,80+2*h)

notes()

def rbf(x, y, c):
	return np.exp( - (np.dot((x - y).T, np.dot(c,(x - y)))))

def HS(data, h, w):

	mean = np.mean(data, 1)

	data = data.T - mean
	
	cov = np.dot(data.T, data) / (data.shape[0] - 1)
	
	data = data + mean
	data = data.T
	N = data.shape[1]
	K = np.zeros((N,N))
	for i in range(N):
		z = data[:,i] - data.T
		K[i,:] = np.exp(-np.sum((z)*((np.dot(cov, z.T).T).T).T, axis=1))
	return K

def create_image(points, w,h, bg=False):
	blur = 3

	image = np.zeros((h,w,3))#np.random.rand(h,w,3) * .5 -.25
	for p in points:
		image[int(p[1]), int(p[0])] = [240.,240.,240.]#240. + np.random.rand(3) * 20. - 10.
	image = cv2.blur(image, (blur,blur))

	grey = cv2.cvtColor(np.copy(image.astype(np.float32)), cv2.COLOR_BGR2GRAY).astype(np.uint8)
	edges = cv2.Canny(np.copy(grey), 50,150)
	cv2.bitwise_not(edges, edges)
	dt = cv2.distanceTransform(np.copy(edges), cv2.DIST_L2, maskSize=5)
		
	cv2.normalize(np.copy(dt), dt, alpha=0., beta=1.0, norm_type=cv2.NORM_MINMAX)# * 10.

	hsv = cv2.cvtColor(np.copy(image.astype(np.float32)), cv2.COLOR_BGR2HSV)
	
	hsv = cv2.blur(np.copy(hsv), (blur,blur))
	dt = cv2.blur(np.copy(dt), (blur,blur))

	return image, hsv, edges, dt

def add_points(points, src, red=False):
	for p in points:
		if red:
			cv2.circle(src, (int(p[0]), int(p[1])), 2, (0,0,255))
		else:
			src[int(p[1]), int(p[0]), :] = [255, 0,0]
	return src

def HSIC(_query, grid, H, N, _KH_, B, row=-1, col=-1):

	query = np.zeros((4, N))

	result = 0

	query[:3,:] = _query[0][grid[0].astype(int),grid[1].astype(int)].T
	query[3,:] = _query[1][grid[0].astype(int),grid[1].astype(int)].T

	L = np.dot(HS(query, _query[0].shape[1]/4, _query[0].shape[0]/4),H)
	_LH_ = np.ravel(L)

	scale = 1. / N
		
	A = scale * np.dot(_KH_, _LH_)
	C = scale * np.dot(_LH_, _LH_)
	result =  A / np.sqrt(B * C)
	# print result,
	return result

# global params
params = util.parameters()
image, hsv, dt, flow = params.get_frames()

# model params
model = ARModel.articulated_model(image.shape)
state = [[0. for i in range(15)], [0. for i in range(15)]]#np.random.rand(dims) * 0.25 - 0.125]
frames = [params.get_frames(frame=1), params.get_frames(frame=2), params.get_frames(frame=3)]

h,w = frames[0][0].shape[:2]

# initialise the reference model
model = ARModel.wiggle(model, state[0])
reference = model.get_points(all_points=True)
model = ARModel.wiggle(model, state[0], inverse=True)

# ----- sweep test ----- #
iterations = 200

state[1][4] -= 0.01 * (iterations/2)
results = np.zeros((4,iterations))

sf = 2
step = 4
h,w = frames[0][0].shape[:2]
h /= sf
w /= sf
y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)

print h/step * w/step
N = y.shape[0]
H = np.eye(N) - (1./N)
# H = H[:400,:]

d1 = [cv2.resize(frames[0][0], (w,h)), cv2.resize(frames[0][2], (w,h))]
d2 = [cv2.resize(frames[1][0], (w,h)), cv2.resize(frames[1][2], (w,h))]
d3 = [cv2.resize(frames[2][0], (w,h)), cv2.resize(frames[2][2], (w,h))]

ref = np.zeros((4, N))
ref[:3,:] = d1[0][[y,x][0].astype(int),[y,x][1].astype(int)].T
ref[3,:] = d1[1][[y,x][0].astype(int),[y,x][1].astype(int)].T
__K = np.dot(HS(ref, d1[0].shape[1]/4, d1[0].shape[0]/4),H)
_KH_1 = np.ravel(__K)
B1 = (1. / N) * np.dot(_KH_1, _KH_1)

ref[:3,:] = d2[0][[y,x][0].astype(int),[y,x][1].astype(int)].T
ref[3,:] = d2[1][[y,x][0].astype(int),[y,x][1].astype(int)].T
__K = np.dot(HS(ref, d2[0].shape[1]/4, d2[0].shape[0]/4),H)
_KH_2 = np.ravel(__K)
B2 = (1. / N) * np.dot(_KH_2, _KH_2)

ref[:3,:] = d3[0][[y,x][0].astype(int),[y,x][1].astype(int)].T
ref[3,:] = d3[1][[y,x][0].astype(int),[y,x][1].astype(int)].T
__K = np.dot(HS(ref, d2[0].shape[1]/4, d2[0].shape[0]/4),H)
_KH_3 = np.ravel(__K)
B3 = (1. / N) * np.dot(_KH_3, _KH_3)

# state[1][2] = 0.#0.005

PSO(state, model, frames, [y,x], sf, w, h, H, N, _KH_2, B2)

# for i in range(iterations):
	
# 	print i,

# 	model = ARModel.wiggle(model, state[1])
# 	query = model.get_points(all_points=True)
# 	image, hsv, edges, dt = create_image(query, w*sf,h*sf, np.copy(frames[0][0]))

# 	image = cv2.resize(image, (w,h))
# 	dt = cv2.resize(dt, (w,h))
# 	display = add_points(query, np.copy(frames[0][0]))

# 	query = model.get_points(all_points=False)
# 	display = add_points(query, np.copy(display), red=True)

# 	display = cv2.resize(display, (w,h))
# 	display[[y,x][0].astype(int),[y,x][1].astype(int)] = [0,255,0]
 
# 	cv2.imshow("Image", image)
# 	cv2.imshow("DT", dt)

# 	cv2.imshow("Image Data", d2[0])
# 	cv2.imshow("DT Data", d2[1])

# 	cv2.imshow("Display", display)

# 	cv2.waitKey(1)

# 	# measure dependence
# 	# results[0][i] = HSIC([image, dt], [y,x], H, N, _KH_1, B1)
# 	results[1][i] = HSIC([image, dt], [y,x], H, N, _KH_2, B2)
# 	# results[2][i] = HSIC([image, dt], [y,x], H, N, _KH_3, B3)
# 	results[3][i] = (results[0][i] + results[1][i] + results[2][i]) / 3.
# 	print results[3][i]
# 	state[1][4] = 0.01

# # plt.show()

# # for i in range(2):
# # r_ = []
# # r_.append(results[0])
# # for i in range(results.shape[0]-2):
# # 	r_.append((results[i] + results[i+2])/2.)
# # plt.plot(r_, ':')
# for i in range(4):
# 	plt.plot(results[i])
# plt.show()









