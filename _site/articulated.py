import numpy as np
import cv2
import util

class joint(object):
	pt 		= np.zeros(4)
	pt_H 	= np.zeros(3)
	pt_2D 	= np.zeros(2)
	R 		= np.eye(4)
	T 		= np.eye(4)

	N = 14
	radius = 110
	points = []

	def __init__(self, _N=14):

		self.N = _N

		self.pt[3] = 1.

		self.points = [np.eye(4) for i in range(self.N)]

		count = 0
		while count < self.N:
			pt = (np.random.rand(2) * 2. * self.radius) - self.radius
			if np.linalg.norm(pt) < self.radius:
				self.points[count][0,3] = pt[0]
				self.points[count][1,3] = pt[1]
				count += 1

	def copy(self, src):
		self.N = src.get_num_points()
		self.radius = src.get_radius()
		self.pt = src.get_pt()
		self.pt_H = src.get_pt_H()
		self.pt_2D = src.get_pt_2D()
		self.R = src.get_R()
		self.T = src.get_T()
		points = src.get_points()
		for i in range(self.N):
				self.points[i] = np.copy(points[i])

	def get_radius(self):
		return self.radius
	def get_pt(self):
		return np.copy(self.pt)
	def get_pt_H(self):
		return np.copy(self.pt_H)
	def get_pt_2D(self):
		return np.copy(self.pt_2D)
	def get_R(self):
		return np.copy(self.R)
	def get_T(self):
		return np.copy(self.T)
	def get_points(self):
		return self.points
	def get_num_points(self):
		return self.N

	def set_pt(self, src):
		self.pt = np.copy(src)
	def set_pt_H(self, src):
		self.pt_H = np.copy(src)
	def set_pt_2D(self, src):
		self.pt_2D = np.copy(src)
	def set_R(self, src):
		self.R = np.copy(src)
	def set_T(self, src):
		self.T = np.copy(src)

class articulated_model(object):

	# general parameters
	num_legs = 2	
	num_joints = 7

	legs = [[]]

	# can be changed to an array for variable lengths later
	segment_length = 68.

	M = np.eye(3)

	def copy(self, src):
		for l in range(self.num_legs):
			for j in range(self.num_joints):
				self.legs[l][j].copy(src.get_legs()[l][j])
	
	def __init__(self, shape, _N=14):

		# initialise camera matrix
		self.M[0,2] = shape[1]/2.
		self.M[1,2] = shape[0]/2.
		self.M[0,0] = 200.
		self.M[1,1] = 200.

		# initialise the legs
		# self.legs = [[] for i in range(self.num_legs)]
		self.legs = [[joint(_N) for i in range(self.num_joints)] for i in range(self.num_legs)]

		for i in range(self.num_legs):
			self.tz(i,1, 1000.)
			self.tx(i,1,-250.)
			self.ty(i,1,250.)
			
			for k in range(2, self.num_joints):
				self.tx(i, k, self.segment_length)	
				self.tx(i, k, self.segment_length)

		self.rz(0,1, - 0.8)
		self.rz(0,2, - 0.4)
		self.rz(0,3, - 0.2)
		
		self.rz(1,1, - 1.96)
		self.rz(1,2, 0.1)

	def get_legs(self):
		return self.legs

	def get_points(self, all_points=False):
		input = []

		for l in range(self.num_legs):
			for p in range(self.num_joints-1, -1, -1):
				
				tr = np.eye(4)										
				
				for k in range(1, p+1):
					tr = np.dot(tr, np.dot(self.legs[l][k].get_T(), self.legs[l][k].get_R())) 
				
				self.legs[l][p].set_pt(np.dot(tr, self.legs[l][0].get_pt()))
				
				if all_points:
					if p <= self.num_joints-1:
						points = self.legs[l][p].get_points()
						for n in range(self.legs[0][1].get_num_points()):
							pt = np.dot(tr, np.dot( points[n] ,self.legs[l][0].get_pt()))
							if p > 0:
								input.append(pt)
				
				else:
					if p > 0:
						input.append(self.legs[l][p].get_pt())

		proj = np.eye(3,4)
		R = np.eye(4)
		T = np.eye(4)
		T[2,3] = 50

		output = []

		for i in range(len(input)):
			H = np.dot(self.M, np.dot(np.eye(3,4), np.dot(T, np.dot(R, input[i]))))
			output.append(H[:2] / H[2])

		return output

	def tx(self, _L, _J, _t):
		T_ = np.eye(4)
		T_[0,3] = _t
		if _L == -1:
			for l in range(self.num_legs):
				self.legs[l][_J].set_T(np.dot(T_, self.legs[l][_J].get_T()))
		else:
			self.legs[_L][_J].set_T(np.dot(T_, self.legs[_L][_J].get_T()))

	def ty(self, _L, _J, _t):
		T_ = np.eye(4)
		T_[1,3] = _t
		if _L == -1:
			for l in range(self.num_legs):
				self.legs[l][_J].set_T(np.dot(T_, self.legs[l][_J].get_T()))
		else:
			self.legs[_L][_J].set_T(np.dot(T_, self.legs[_L][_J].get_T()))

	def tz(self, _L, _J, _t):
		T_ = np.eye(4)
		T_[2,3] = _t
		if _L == -1:
			for l in range(self.num_legs):
				self.legs[l][_J].set_T(np.dot(T_, self.legs[l][_J].get_T()))
		else:
			self.legs[_L][_J].set_T(np.dot(T_, self.legs[_L][_J].get_T()))

	def rx(self, _L, _J, _r):
		R_ = np.eye(4)
		R_[1,1] = np.cos(_r)
		R_[2,2] = np.cos(_r)
		R_[1,2] = - np.sin(_r)
		R_[2,1] = np.sin(_r)
		if _L == -1:
			for l in range(self.num_legs):
				self.legs[l][_J].set_R(np.dot(R_, self.legs[l][_J].get_R()))
		else:
			self.legs[_L][_J].set_R(np.dot(R_, self.legs[_L][_J].get_R()))

	def ry(self, _L, _J, _r):
		R_ = np.eye(4)
		R_[0,0] = np.cos(_r)
		R_[2,2] = np.cos(_r)
		R_[0,2] = np.sin(_r)
		R_[2,0] = - np.sin(_r)
		if _L == -1:
			for l in range(self.num_legs):
				self.legs[l][_J].set_R(np.dot(R_, self.legs[l][_J].get_R()))
		else:
			self.legs[_L][_J].set_R(np.dot(R_, self.legs[_L][_J].get_R()))

	def rz(self, _L, _J, _r):
		R_ = np.eye(4)
		R_[0,0] = np.cos(_r)
		R_[1,1] = np.cos(_r)
		R_[0,1] = - np.sin(_r)
		R_[1,0] = np.sin(_r)
		if _L == -1:
			for l in range(self.num_legs):
				self.legs[l][_J].set_R(np.dot(R_, self.legs[l][_J].get_R()))
		else:
			self.legs[_L][_J].set_R(np.dot(R_, self.legs[_L][_J].get_R()))



def optimise(state, initial_model, params, P_KH, HS, display=False):
	image, hsv, dt = params.get_frames()
	model = articulated_model(image.shape)

	model.copy(initial_model)

	## move the model into the given pose
	model.tx(-1,1,state[0])
	model.tx(-1,1,state[1])
	model.rz(-1,1,state[2])

	model.rz(0,1,state[3])
	model.rz(1,1,state[4])
	
	if display:
		## draw the model for testing
		points = model.get_points()
		for i in range(len(points)):
			cv2.circle(image,(int(points[i][0]),int(points[i][1])),2,(0,0,255))
		for i in range(len(points)-1):
			if i != 5:
				cv2.line(image,(int(points[i][0]),int(points[i][1])),(int(points[i+1][0]),int(points[i+1][1])),(255,0,0))
		
		points = model.get_points(all_points=True)
		for i in range(len(points)):
			cv2.circle(image,(int(points[i][0]),int(points[i][1])),1,(0,255,0))

		__dir, __counter, __current_frame, __suffix = params.get_filename() 

		cv2.imwrite(__dir + str(__counter) + '__' + str(__current_frame) + __suffix, image)

		cv2.imshow("Neptune - Image", image)
		cv2.waitKey(1)

	## get the points and HS matrix
	points = model.get_points(all_points=True)
	data, LH = util.descriptors(params, points, HS)

	if np.sum(data) == 0:
		return 0.

	N = HS.get_N()

	result = ((1. / (N*N)) * np.trace(np.dot(P_KH,LH))) / (np.sqrt((1. / (N*N)) * np.trace(np.dot(LH,LH))))
	
	print state, result

	return result

def PSO(state, initial_model, params, P_KH, HS):

	M = 20        # number of particles
	K = len(state) 	      # num parameters to define functions

	c1, c2, omega       = 1.49618, 1.49618, 0.7298
	p, v = np.zeros((M, K+1)), np.zeros((M, K))
	b, g = np.zeros((M, K+1)), np.zeros(K+1)

	MAX_ITER = 10

	results = []

	for i in range(M):	
		p[i,0]   = (np.random.rand(1) * 10.) - 5.
		p[i,1]   = (np.random.rand(1) * 10.) - 5.
		p[i,2]   = (np.random.rand(1) * 0.5) - 0.25
		p[i,3]   = (np.random.rand(1) * 0.5) - 0.25
		p[i,4]   = (np.random.rand(1) * 0.5) - 0.25
		p[i,K]   = optimise(p[i,:K], initial_model, params, P_KH, HS)
		v[i,0]   = (np.random.rand(1) * 5.) - 2.5
		v[i,1]   = (np.random.rand(1) * 5.) - 2.5
		v[i,2]   = (np.random.rand(1) * 0.25) - 0.125
		v[i,3]   = (np.random.rand(1) * 0.25) - 0.125
		v[i,4]   = (np.random.rand(1) * 0.25) - 0.125
		b[i] = np.copy(p[i])

 #    # find the global opt
 	max, idx, = np.copy(p[0,K]), 0
 	for i in range(1, M):
 		if p[i,K] > max:
 			max, idx = np.copy(p[i,K]), i
 	g = np.copy(p[idx,:])

	# ## ==== RUN PSO ==== ##
	for iter in range(MAX_ITER):

		print '------------>', iter, g[K]

		for i in range(M):
			p[i,K]      = optimise(p[i,:K], initial_model, params, P_KH, HS)
			if p[i,K] > b[i,K]:
				b[i,:] = np.copy(p[i,:])

		max, idx = np.copy(g[K]), -1
		for i in range(0, M):
			if p[i,K] > max:
				max, idx = np.copy(p[i,K]), i

		if idx > -1:
			g = np.copy(p[idx,:])

		optimise(g[:K], initial_model, params, P_KH, HS, display=False)

		for i in range(M):
			r1, r2 = np.random.rand(1)[0], np.random.rand(1)[0]
			v[i] = (omega * np.copy(v[i])) + (c1 * r1 * (np.copy(b[i,:K]) - np.copy(p[i,:K]))) + (c2 * r2 * (np.copy(g[:K]) - np.copy(p[i,:K])))
			p[i,:K] += np.copy(v[i])


	optimise(g[:K], initial_model, params, P_KH, HS, display=True)






