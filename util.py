import numpy as np
import cv2
import HSIC

# # dir = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/ball/ball'
# dir = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CHG/0008/0002/frame-000'
# current_frame = 0
# position = np.zeros(3)
# position[:] = 100,100,-1
# radius = 52
# file = dir + str(current_frame) + '.jpg'
# __image__ = cv2.imread(file, cv2.IMREAD_COLOR)

# def get_features(N):
# 	features = np.zeros((N,2))
# 	count = 0
# 	while count < N:
# 		offset = ((np.random.rand(2) * 2*(radius-1)) - radius).astype(int)
# 		if np.linalg.norm(offset) < radius:
# 			features[count,:] = offset
# 			count += 1
# 	return features

# features = [get_features(HSIC.N) for i in range(4)]

# def initialise(image):

# 	positive = []
# 	negative = []

# 	delta = np.zeros(3)
# 	delta[:] = 0.,0.,0.

# 	p = model()
# 	p.init(image, features[0])	
# 	positive.append(p)

# 	n = model()
# 	n.init(image, features[0])
# 	negative.append(n)

# 	for i in range(len(features)-1):
# 		p = model()
# 		p.set(positive[0].state, image, features[i+1])
# 		# p.update(delta, image, features[i+1], True)
# 		positive.append(p)

# 		# p.show_state(image, features[i+1])

# 		n = model()
# 		n.set(negative[0].state+delta, image, features[i+1])
# 		# n.update(delta, image, features[i+1], True)
# 		negative.append(n)

# 		# n.show_state(image, features[i+1])

# 	return positive, negative

# def get_descriptors(image, state, features, N, K):

# 	descriptor = np.zeros((K,N))

# 	pose = ((state[2] * np.copy(features)) + np.copy(state[:2])).astype(int)

# 	grey = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2GRAY)
# 	edges = cv2.Canny(np.copy(grey), 50,150)

# 	edges = cv2.bitwise_not(np.copy(edges))
 
# 	dt = cv2.distanceTransform(np.copy(edges), cv2.DIST_L2, maskSize=5)

# 	cv2.normalize(np.copy(dt), dt, alpha=0., beta=1.0, norm_type=cv2.NORM_MINMAX) * 10.

# 	hsv = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2HSV)

# 	image = cv2.blur(np.copy(image), (5,5))
# 	hsv = cv2.blur(np.copy(hsv), (5,5))
# 	dt = cv2.blur(np.copy(dt), (5,5))

# 	# cv2.imshow("Edges", edges)
# 	# cv2.imshow("Neptune", image)
# 	# cv2.imshow("DT", dt)
# 	# cv2.imshow("hsv", hsv)
# 	# command = cv2.waitKey(1)

# 	for i in range(N):

# 		descriptor[0,i] = hsv[pose[i,0],pose[i,1],0]
# 		descriptor[1,i] = hsv[pose[i,0],pose[i,1],1]
# 		descriptor[2,i] = hsv[pose[i,0],pose[i,1],2]

# 		descriptor[3,i] = image[pose[i,0],pose[i,1],0]
# 		descriptor[4,i] = image[pose[i,0],pose[i,1],1]
# 		descriptor[5,i] = image[pose[i,0],pose[i,1],2]

# 		descriptor[6,i] = dt[pose[i,0],pose[i,1]]

# 	return descriptor

# def get_state():

# 	global dir, position, file

# 	image = cv2.imread(file, cv2.IMREAD_COLOR)

# 	cv2.namedWindow('Neptune')
# 	cv2.setMouseCallback('Neptune',draw_rect)

# 	command = cv2.waitKey(1)
# 	state = np.zeros(3)
# 	state[:] = -1,-1,1.0

# 	while state[0] == -1:

# 		image = cv2.imread(file, cv2.IMREAD_COLOR)
# 		cv2.circle(image,(int(position[1]),int(position[0])),radius,(0,0,255))

# 		if position[2] != -1:
# 			state[:2] = position[:2]

# 		cv2.imshow("Neptune", image)
# 		command = cv2.waitKey(1)

# 	position[:] = 100,100,-1

# 	return state

# def draw_rect(event,x,y,flags,param):

# 	position[0] = y
# 	position[1] = x
# 	if event == cv2.EVENT_LBUTTONDOWN:
# 		position[2] = 0

# class model():

# 	state = np.zeros((3))
# 	state[:] = -1,-1,1.0
# 	data = []
# 	KH = np.zeros((HSIC.N,HSIC.N))

# 	def init(self, image, _F):	
# 		self.state = get_state()
# 		self.data = get_descriptors(image, self.state, _F, HSIC.N, HSIC.K)

# 		for i in range(HSIC.N):
# 			for k in range(HSIC.N):
# 				self.KH[i,k] = HSIC.rbf(self.data[:,i], self.data[:,k], HSIC.S)

# 		self.KH = np.dot(self.KH,HSIC.H)

# 	def show_state(self, _F):
# 		# image = np.copy(image)
# 		_image = cv2.imread(file, cv2.IMREAD_COLOR)
# 		pose = ((self.state[2] * np.copy(_F)) + np.copy(self.state[:2])).astype(int)

# 		cv2.circle(_image,(int(self.state[1]),int(self.state[0])),int(radius*self.state[2]),(0,0,255), 2)

# 		# for i in range(HSIC.N):
# 			# cv2.circle(_image,(int(pose[i,1]),int(pose[i,0])),1,(255,0,0), -1)

# 		print '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/results/ball/'+str(current_frame)+'.png'
# 		cv2.imwrite('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/results/ball/04/'+str(current_frame)+'.png', _image)

# 		# cv2.imshow("Neptune", _image)
# 		# cv2.waitKey(1)

# 	def set(self, src, image, _F):
# 		self.state = np.copy(src)#np.copy(src.state)
# 		self.data = get_descriptors(image, self.state, _F, HSIC.N, HSIC.K)

# 		for i in range(HSIC.N):
# 			for k in range(HSIC.N):
# 				self.KH[i,k] = HSIC.rbf(self.data[:,i], self.data[:,k], HSIC.S)

# 		self.KH = np.dot(self.KH,HSIC.H)


class parameters(object):
	output_dir = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CHG/output/'
	dir = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CHG/set1/0008/0002/frame-'
	# dir = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CHG/set3/0004/0012/frame-'
	suffix = '.jpg'
	current_frame = 29
	file = dir + str(current_frame).zfill(4) + suffix
	image = cv2.imread(file, cv2.IMREAD_COLOR)

	h,w = image.shape[:2]

	grey = np.zeros((h,w,1))
	edges = np.zeros((h,w,1))
	dt = np.zeros((h,w,1))
	hsv = np.zeros((h,w,3))

	counter = 15

	noise_counter = 0

	def get_filename(self):
		self.counter += 1
		return self.output_dir, self.counter, self.current_frame, self.suffix

	def __init__(self):
		self.current_frame = -1
		self.update()
		# self.current_frame = 28

	def update(self):

		self.current_frame += 1

		self.file = self.dir + str(self.current_frame).zfill(4) + self.suffix
		
		self.image = cv2.imread(self.file, cv2.IMREAD_COLOR)
		
		self.grey = cv2.cvtColor(np.copy(self.image), cv2.COLOR_BGR2GRAY)
		self.edges = cv2.Canny(np.copy(self.grey), 50,150)
		self.dt = cv2.distanceTransform(np.copy(self.edges), cv2.DIST_L2, maskSize=5)
		
		cv2.normalize(np.copy(self.dt), self.dt, alpha=0., beta=1.0, norm_type=cv2.NORM_MINMAX) * 10.
		
		self.hsv = cv2.cvtColor(np.copy(self.image), cv2.COLOR_BGR2HSV)
		
		self.image = cv2.blur(np.copy(self.image), (5,5))
		self.hsv = cv2.blur(np.copy(self.hsv), (5,5))
		self.dt = cv2.blur(np.copy(self.dt), (5,5))

	def get_frames(self):
		return np.copy(self.image), np.copy(self.hsv), np.copy(self.dt)

	def apply_noise(self):

		self.current_frame -= 1
		self.update()

		# num = 40000

		# x = [(np.random.rand(num)*self.image.shape[0]).astype(int)]
		# y = [(np.random.rand(num)*self.image.shape[1]).astype(int)]
		# r1 = [np.random.rand(3) * 1000. for i in range(num)]
		# r2 = [np.random.rand(3) * 1000. for i in range(num)]
		# r3 = [np.random.rand() * 1000. for i in range(num)]

		# self.image[x, y, :] = r1 	#np.random.rand(3) * 1000.
		# self.hsv[x, y, :] 	= r2 	#np.random.rand(3) * 1000.
		# self.dt[x, y] 		= r3 	#np.random.rand() * 1000.

		x = 90
		y = 80
		x_ = 20
		y_ = 60

		for r in range(y - y_, y + y_):
			for c in range(x - x_, y + x_):
				self.image[r,c,:] = np.random.rand(3) * 10.
				self.hsv[r,c,:] = np.random.rand(3) * 10.
				self.dt[r,c] = np.random.rand()

		# cv2.imwrite('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CHG/pose_results/'+str(self.noise_counter)+'.png', self.image)
		self.noise_counter += 1

def descriptors(params, points, HS, init=False):

	data = np.zeros((HS.get_K(), len(points)))
	KH = np.zeros((HS.get_N(),HS.get_N()))

	image_, hsv_, dt_ = params.get_frames()

	# for p in points:
	# 	if (p[0] < 0) or (p[1] < 0) or (p[0] >= image_.shape[1]-10) or (p[1] >= image_.shape[0]-10):
	# 		data[:,:] = 0
	# 		return data, KH
	# 	if p[0] >= image_.shape[1]-10:
	# 		data[:,:] = 0
	# 		return data, KH
	# 	if p[1] >= image_.shape[0]-10:
	# 		data[:,:] = 0
	# 		return data, KH

	idx = 0
	for p in points:
		if ((p[0] < 0) or (p[1] < 0)
			or (p[0] >= image_.shape[1]-10) 
				or (p[1] >= image_.shape[0]-10)
					or (p[0] >= image_.shape[1]-10)
						or (p[1] >= image_.shape[0]-10)):
							data[:6,idx] = np.random.rand() * 255
							data[6,idx] = np.random.rand() * 20.
		else:
			data[0,idx] = hsv_[points[idx][1],points[idx][0],0]
			data[1,idx] = hsv_[points[idx][1],points[idx][0],1]
			data[2,idx] = hsv_[points[idx][1],points[idx][0],2]

			data[3,idx] = image_[points[idx][1],points[idx][0],0]
			data[4,idx] = image_[points[idx][1],points[idx][0],1]
			data[5,idx] = image_[points[idx][1],points[idx][0],2]

			data[6,idx] = dt_[points[idx][1],points[idx][0]]
		idx += 1


	# for i in range(HS.get_N()):

	# 	data[0,i] = hsv_[points[i][1],points[i][0],0]
	# 	data[1,i] = hsv_[points[i][1],points[i][0],1]
	# 	data[2,i] = hsv_[points[i][1],points[i][0],2]

	# 	data[3,i] = image_[points[i][1],points[i][0],0]
	# 	data[4,i] = image_[points[i][1],points[i][0],1]
	# 	data[5,i] = image_[points[i][1],points[i][0],2]

	# 	data[6,i] = dt_[points[i][1],points[i][0]]


	if init:
		HS.set_covariance(data)

	# else:
	for i in range(HS.get_N()):
		for k in range(HS.get_N()):
			KH[i,k] = HS.rbf(data[:,i], data[:,k])

	KH = np.dot(KH,HS.get_H())

	return data, KH








