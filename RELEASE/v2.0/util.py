import cv2
import numpy as np

class parameters(object):
	output_dir = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CHG/output/'
	# dir = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CHG/set1/0008/0003/frame-'
	dir = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CHG/set3/0004/0012/frame-'
	suffix = '.jpg'

	counter = -1
	blur = 3

	def get_filename(self):
		self.counter += 1
		return self.output_dir, self.counter, self.current_frame, self.suffix

	def __init__(self):
		self.current_frame = 0
		self.update()

	def update(self):

		self.current_frame += 1

		self.file = self.dir + str(self.current_frame).zfill(4) + self.suffix
		
		self.image = cv2.imread(self.file, cv2.IMREAD_COLOR)

		self.h,self.w = self.image.shape[:2]

		self.prev_frame = cv2.imread(self.dir + str(self.current_frame-1).zfill(4) + self.suffix, 0)
		self.next_frame = cv2.imread(self.dir + str(self.current_frame).zfill(4) + self.suffix,0)

		self.flow = cv2.calcOpticalFlowFarneback(self.prev_frame,self.next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

		self.grey = cv2.cvtColor(np.copy(self.image), cv2.COLOR_BGR2GRAY)
		self.edges = cv2.Canny(np.copy(self.grey), 50,150)
		self.dt = cv2.distanceTransform(np.copy(self.edges), cv2.DIST_L2, maskSize=5)
		
		cv2.normalize(np.copy(self.dt), self.dt, alpha=0., beta=1.0, norm_type=cv2.NORM_MINMAX) * 10.
		
		self.hsv = cv2.cvtColor(np.copy(self.image), cv2.COLOR_BGR2HSV)
		
		self.image = cv2.blur(np.copy(self.image), (self.blur,self.blur))
		self.hsv = cv2.blur(np.copy(self.hsv), (self.blur,self.blur))
		self.dt = cv2.blur(np.copy(self.dt), (self.blur,self.blur))

	def get_frames(self):
		return np.copy(self.image), np.copy(self.hsv), np.copy(self.dt), np.copy(self.flow)


def descriptors(params, points, prev_points, HS, _train=False, _motion=False):

	KH = np.zeros((HS.get_N(),HS.get_N()))

	image_, hsv_, dt_, flow_ = params.get_frames()

	data = np.zeros((HS.get_K(), HS.get_N()))

	num_out = 0

	idx = 0
	for p in points:
		if ((p[0] < 0) or (p[1] < 0) or (p[0] >= image_.shape[1]-10) 
			or (p[1] >= image_.shape[0]-10) or (p[0] >= image_.shape[1]-10) or (p[1] >= image_.shape[0]-10)):
					data[:, idx] = np.random.rand(data.shape[0])
					num_out += 1
		else:

			data[ :3, idx] 	= hsv_[points[idx][1],points[idx][0],:]
			data[3:6, idx] 	= image_[points[idx][1],points[idx][0],:]
			data[6,idx] 	= dt_[points[idx][1],points[idx][0]]

			if _motion:
				if _train:
					data[7:,idx] = flow_[prev_points[idx][1],prev_points[idx][0],:]
				else:
					data[7,idx] = prev_points[idx][0] - points[idx][0]
					data[8,idx] = prev_points[idx][1] - points[idx][1]

		idx += 1

	if num_out > len(points) / 2:
		KH = np.random.rand(HS.get_N(),HS.get_N()) * 0.5
		return KH

	HS.set_covariance(data)

	cov = HS.get_cov()

	for i in range(HS.get_N()):
		KH[i,:] = np.exp(-np.sum((data[:,i] - data.T)*((np.dot(cov, (data[:,i] - data.T).T).T).T).T, axis=1))

	KH = np.dot(KH,HS.get_H())

	return KH



