import numpy as np
import model as ARModel
import util
import cv2
import matplotlib.pyplot as plt

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

def get_image_data(f, grid, H, N):
	ref = np.zeros((4, N))
	ref[:3,:] = f[0][grid[0].astype(int),grid[1].astype(int)].T
	ref[3,:] = f[2][grid[0].astype(int),grid[1].astype(int)].T
	return np.ravel(np.dot(HS(ref),H))

def HS(data):

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

def HSIC(model, state, frames, w, h, sf, grid, H, N, K, B):

	model = ARModel.wiggle(model, state)
	query = model.get_points(all_points=True)

	image, hsv, edges, dt = create_image(query, w*sf,h*sf, np.copy(frames[0]))
	model = ARModel.wiggle(model, state, inverse=True)
	
	image, dt = cv2.resize(image, (w,h)), cv2.resize(dt, (w,h))

	data = np.zeros((4, N))

	result = 0
	data[:3,:] = image[grid[0].astype(int),grid[1].astype(int)].T
	data[3,:]  = dt[grid[0].astype(int),grid[1].astype(int)].T

	L = np.ravel(np.dot(HS(data),H))

	scale = 1. / N
		
	A = scale * np.dot(K, L)
	C = scale * np.dot(L, L)
	result =  A / np.sqrt(B * C)
	return result

def PSO(_state, _model, _frames, _grid, _sf, _w, _h, _H, _N, _K, _B):

	M = 50      # number of particles
	K = len(_state) 	      # num parameters to define functions

	c1, c2, omega       = 1.49618, 1.49618, 0.7298
	p, v = np.zeros((M, K+1)), np.zeros((M, K))
	b, g = np.zeros((M, K+1)), np.zeros(K+1)

	MAX_ITER = 1

	results = []

	p[0,:K] = _state
	p[0,K] = HSIC(_model, p[0,:], _frames,  _w, _h, _sf, _grid, _H, _N, _K, _B)

	for i in range(1, M):	
		
		p[i,:K] = _state + ((np.random.rand(K) * 0.4) - 0.2)
		p[i,:2] *= 200.

		v[i,:]  = ((np.random.rand(K) * 0.2) - 0.1)
		v[i,:2] *= 20.

		p[i,K] = HSIC(_model, p[i,:], _frames,  _w, _h, _sf, _grid, _H, _N, _K, _B)
		b[i] = np.copy(p[i])

 	_max, idx, = np.copy(p[0,K]), 0
 	for i in range(1, M):
 		if p[i,K] > _max:
 			_max, idx = np.copy(p[i,K]), i
 	g = np.copy(p[idx,:])

	# ## ==== RUN PSO ==== ##
	for iter in range(MAX_ITER):

		print '-->', iter, g[K]

		for i in range(M):
			p[i,K] = HSIC(_model, p[i,:], _frames, _w, _h, _sf, _grid, _H, _N, _K, _B)
			if p[i,K] > b[i,K]:
				b[i,:] = np.copy(p[i,:])

		_max, idx = np.copy(g[K]), -1
		for i in range(0, M):
			if p[i,K] > _max:
				_max, idx = np.copy(p[i,K]), i

		if idx > -1:
			g = np.copy(p[idx,:])

		# this call is just to draw the current global opt to the screen. The value is computed / copied earlier
		# g[K] = HSIC(model, g, w*sf, h*sf, grid, H, N, K, B)

		for i in range(M):
			r1, r2 = np.random.rand(1)[0], np.random.rand(1)[0]
			v[i] = (omega * np.copy(v[i])) + (c1 * r1 * (np.copy(b[i,:K]) - np.copy(p[i,:K]))) + (c2 * r2 * (np.copy(g[:K]) - np.copy(p[i,:K])))
			p[i,:K] += np.copy(v[i])

		_state = np.copy(g[:K])

	return _state

# global params
params = util.parameters()
image, hsv, dt, flow = params.get_frames()

t = [1, 6, 12]
M = len(t)

# model params
model = ARModel.articulated_model(image.shape)
# state = [[0. for i in range(15)] for i in range(M)]
state = [np.random.rand(15) * 0.25 - 0.125 for i in range(M)]
frames = [params.get_frames(frame=t[i]) for i in range(M)]

# ----- sweep test ----- #
sf, step = 2, 4
h,w = frames[0][0].shape[:2]
h,w = h/sf, w/sf
y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)

N = y.shape[0]
H = np.eye(N) - (1./N)

f = [[cv2.resize(frames[k][i], (w,h)) for i in range(4) ] for k in range(M)]

K = [get_image_data(f[k], [y,x], H, N) for k in range(M)]
B = [(1. / N) * np.dot(K[i], K[i]) for i in range(M)]

for i in range(M):
	state[i] = PSO(state[i], model, f[i], [y,x], sf, w, h, H, N, K[i], B[i])

	print 'Output: ', state[i][:4]

for k in range(M-1):

	for i in range(1, t[k+1] - t[k] + 2):

		display_frame = params.get_frames(frame=i+k)

		temp_state = state[k] + ((state[k+1] - state[k]) / (t[k+1]-t[k]) * (i-1))
		print k, i, temp_state[:4]

		model = ARModel.wiggle(model, temp_state)
		query = model.get_points(all_points=True)
		model = ARModel.wiggle(model, temp_state, inverse=True)

		display = add_points(query, np.copy(display_frame[0]))
		# display[[y,x][0].astype(int),[y,x][1].astype(int)] = [0,255,0]

		cv2.imshow("PSO", display)
		cv2.waitKey(0)



