import numpy as np

class joint(object):
	pt 		= np.zeros(4)
	pt_H 	= np.zeros(3)
	pt_2D 	= np.zeros(2)
	R 		= np.eye(4)
	T 		= np.eye(4)

	N = 10
	radius = 200
	points = []

	def __init__(self):
		self.pt[3] = 1.

		self.points = [np.eye(4) for i in range(self.N)]

		count = 0
		while count < self.N:
			pt = (np.random.rand(2) * 2. * self.radius) - self.radius
			if np.linalg.norm(pt) < self.radius:
				self.points[count][0,3] = pt[0]
				self.points[count][1,3] = pt[1]
				count += 1

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
		return np.copy(self.points)
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
	num_joints = 6

	legs = [[]]

	# can be changed to an array for variable lengths later
	segment_length = 180.

	M = np.eye(3)
	
	def __init__(self, shape):

		# initialise camera matrix
		self.M[0,2] = shape[1]/2.
		self.M[1,2] = shape[0]/2.

		# initialise the legs
		# self.legs = [[] for i in range(self.num_legs)]
		self.legs = [[joint() for i in range(self.num_joints)] for i in range(self.num_legs)]

		for i in range(self.num_legs):
			self.tz(i,1,10.)
			self.tx(i,1,-500.)
			self.ty(i,1,500.)
			
			for k in range(2, self.num_joints):
				self.tx(i, k, self.segment_length)	
				self.tx(i, k, self.segment_length)

		self.rz(0,1, - 0.8)
		self.rz(0,2, - 0.4)
		self.rz(0,3, - 0.2)
		
		self.rz(1,1, - 1.96)
		self.rz(1,2, 0.1)

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

		output = []

		for i in range(len(input)):
			H = np.dot(self.M, np.dot(np.eye(3,4), np.dot(T, np.dot(R, input[i]))))
			output.append(H[:2] / H[2])

		return output

	def tx(self, _L, _J, _t):
		T_ = np.eye(4)
		T_[0,3] = _t
		self.legs[_L][_J].set_T(np.dot(T_, self.legs[_L][_J].get_T()))

	def ty(self, _L, _J, _t):
		T_ = np.eye(4)
		T_[1,3] = _t
		self.legs[_L][_J].set_T(np.dot(T_, self.legs[_L][_J].get_T()))

	def tz(self, _L, _J, _t):
		T_ = np.eye(4)
		T_[2,3] = _t
		self.legs[_L][_J].set_T(np.dot(T_, self.legs[_L][_J].get_T()))

	def rx(self, _L, _J, _r):
		R_ = np.eye(4)
		R_[1,1] = np.cos(_r)
		R_[2,2] = np.cos(_r)
		R_[1,2] = - np.sin(_r)
		R_[2,1] = np.sin(_r)
		self.legs[_L][_J].set_R(np.dot(R_, self.legs[_L][_J].get_R()))

	def ry(self, _L, _J, _r):
		R_ = np.eye(4)
		R_[0,0] = np.cos(_r)
		R_[2,2] = np.cos(_r)
		R_[0,2] = np.sin(_r)
		R_[2,0] = - np.sin(_r)
		self.legs[_L][_J].set_R(np.dot(R_, self.legs[_L][_J].get_R()))

	def rz(self, _L, _J, _r):
		R_ = np.eye(4)
		R_[0,0] = np.cos(_r)
		R_[1,1] = np.cos(_r)
		R_[0,1] = - np.sin(_r)
		R_[1,0] = np.sin(_r)
		self.legs[_L][_J].set_R(np.dot(R_, self.legs[_L][_J].get_R()))

















