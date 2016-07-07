import numpy as np
import model as ARModel
import util
import cv2
import matplotlib.pyplot as plt


def notes():

	print ''' 

1. Define a grid of points over each joint and then take those that lie within
	   the defined ellipse.

2. HSIC - only take the k nearest neighbours when constructing K / L etc
			- i.e. so that each point is only compared to a local region.

'''

w, h = 320, 240

cv2.namedWindow("Image")
cv2.moveWindow("Image", 10,10)

cv2.namedWindow("Edges")
cv2.moveWindow("Edges", 20+w,10)

cv2.namedWindow("DT")
cv2.moveWindow("DT", 30+2*w,10)

cv2.namedWindow("HSV")
cv2.moveWindow("HSV", 40+3*w,10)

cv2.namedWindow("Image Data")
cv2.moveWindow("Image Data", 10,50+h)

cv2.namedWindow("Edges Data")
cv2.moveWindow("Edges Data", 20+w,50+h)

cv2.namedWindow("DT Data")
cv2.moveWindow("DT Data", 30+2*w,50+h)

cv2.namedWindow("HSV Data")
cv2.moveWindow("HSV Data", 40+3*w,50+h)

cv2.namedWindow("Display")
cv2.moveWindow("Display", 10,80+2*h)

notes()

def HS(data):

	mean = np.mean(data, 1)

	for i in range(data.shape[1]):
		data[:,i] -= mean

	cov = np.dot(data, data.T) / (data.shape[1] - 1)

	for i in range(data.shape[1]):
		data[:,i] += mean

	# project into inner product space
	N = data.shape[1]
	K = np.zeros((N,N))
	for i in range(N):
		K[i,:] = np.exp(-np.sum((data[:,i] - data.T)*((np.dot(cov, (data[:,i] - data.T).T).T).T).T, axis=1))

	return K

def create_image(points, w,h):
	# image = np.ones((h,w,3))*240.#np.random.rand(h,w,3) * .5 -.25
	# for p in points:
	# 	image[int(p[1]), int(p[0])] = [10.,10.,10.]#240. + np.random.rand() * 2. - 1.
	# image = cv2.blur(image, (11,11))
	# return image

	blur = 7

	image = np.zeros((h,w,3))#np.random.rand(h,w,3) * .5 -.25
	for p in points:
		image[int(p[1]), int(p[0])] = [240.,240.,240.]#240. + np.random.rand(3) * 20. - 10.
	image = cv2.blur(image, (blur,blur))

	grey = cv2.cvtColor(np.copy(image.astype(np.float32)), cv2.COLOR_BGR2GRAY).astype(np.uint8)
	edges = cv2.Canny(np.copy(grey), 50,150)
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

def HSIC(_ref, _query, grid, H, N, row=-1, col=-1):

	# _ref[0] = _ref.astype(float)

	ref = np.zeros((4, N))
	query = np.zeros((4, N))

	result = 0

	if row > -1:
		ref = _ref[row,:220].T
		query = _query[row,:220].T
		N = ref.shape[1]
		H = np.eye(N) - (1./N)

	else:
		ref[:3,:] = _ref[0][grid[0].astype(int),grid[1].astype(int)].T
		query[:3,:] = _query[0][grid[0].astype(int),grid[1].astype(int)].T

		ref[3,:] = _ref[1][grid[0].astype(int),grid[1].astype(int)].T
		query[3,:] = _query[1][grid[0].astype(int),grid[1].astype(int)].T

		# ref[4:,:] = _ref[2][grid[0].astype(int),grid[1].astype(int)].T
		# query[4:,:] = _query[2][grid[0].astype(int),grid[1].astype(int)].T

	K, L = np.dot(HS(ref),H), np.dot(HS(query),H)

	_KH_ = np.ravel(K)
	_LH_ = np.ravel(L)

	scale = 1. / N
		
	A = scale * np.dot(_KH_, _LH_)
	B = scale * np.dot(_KH_, _KH_)
	C = scale * np.dot(_LH_, _LH_)

	result =  A / np.sqrt(B * C)
	print result
	return result

# global params
params = util.parameters()
image, hsv, dt, flow = params.get_frames()

# model params
model = ARModel.articulated_model(image.shape)
state = [[0. for i in range(15)], [0. for i in range(15)]]#np.random.rand(dims) * 0.25 - 0.125]
frames = [params.get_frames(10), params.get_frames(0)]

h,w = frames[0][0].shape[:2]

# initialise the reference model
model = ARModel.wiggle(model, state[0])
reference = model.get_points(all_points=True)
model = ARModel.wiggle(model, state[0], inverse=True)

# ----- sweep test ----- #
iterations = 120
# state[1][2] += 0.005 * 60

# # move model
# model = ARModel.wiggle(model, state[1])
# query = model.get_points(all_points=True)
# model = ARModel.wiggle(model, state[1], inverse=True)

# # get data
# image, hsv, edges, dt = create_image(query, w,h)

# display = add_points(query, np.copy(frames[0][0]))
# cv2.imshow("Display", display)

# cv2.waitKey(0)

state[1][2] -= 0.005 * (iterations/2)
results = np.zeros(iterations)

sf = 1

step = 6
h,w = frames[0][0].shape[:2]
h /= sf
w /= sf
y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)

N = y.shape[0]
H = np.eye(N) - (1./N)

for i in range(iterations):
	
	print i,

	# move model
	model = ARModel.wiggle(model, state[1])
	query = model.get_points(all_points=True)
	model = ARModel.wiggle(model, state[1], inverse=True)

	# get data
	image, hsv, edges, dt = create_image(query, w*sf,h*sf)

	image = cv2.resize(image, (w,h))
	dt = cv2.resize(dt, (w,h))

	data_im = cv2.resize(frames[0][0], (w,h))
	data_dt = cv2.resize(frames[0][2], (w,h))


	display = add_points(query, np.copy(frames[0][0]))

	model = ARModel.wiggle(model, state[1])
	query = model.get_points(all_points=False)
	model = ARModel.wiggle(model, state[1], inverse=True)
	display = add_points(query, np.copy(display), red=True)


	display = cv2.resize(display, (w,h))
	display[[y,x][0].astype(int),[y,x][1].astype(int)] = [0,255,0]

	# plt.plot(image[100,:])
	# plt.plot(frames[0][0][100,:])

	# plt.show()

	cv2.imshow("Image", image)
	cv2.imshow("Edges", edges)
	cv2.imshow("DT", dt)
	cv2.imshow("HSV", hsv)

	cv2.imshow("Image Data", data_im)
	cv2.imshow("Edges Data", frames[0][3])
	cv2.imshow("DT Data", data_dt)
	cv2.imshow("HSV Data", frames[0][1])

	cv2.imshow("Display", display)

	cv2.waitKey(1)

	# measure dependence
	results[i] = HSIC([data_im, data_dt, frames[0][1]], [image, dt, hsv], [y,x], H, N)

	state[1][2] += 0.005

# plt.show()

# for i in range(2):
# r_ = []
# r_.append(results[0])
# for i in range(results.shape[0]-2):
# 	r_.append((results[i] + results[i+2])/2.)
# plt.plot(r_, ':')
plt.plot(results)
plt.show()









