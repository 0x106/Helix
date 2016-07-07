

import model as ARModel
import HSIC
import util
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

print 'here'

descriptors = []

def test(model):
	# model = ARModel.wiggle(model, ((np.random.rand(dims) * 0.125) - 0.0625))

	command = cv2.waitKey(1)
	num = 60

	results = [np.zeros(num) for i in range(10)]

	mean, cov = [np.zeros(dims), np.zeros(dims)],[np.zeros((dims,dims)), np.zeros((dims,dims))]

	for frame in range(10):

		P_KH = ARModel.get_KH(model, [0. for i in range(dims)], params, HS)
		[params.update() for i in range(1)]

		model.rz(-1,1,-(num/2)*0.01)

		points = model.get_points(all_points=True)
		util.descriptors(params, points, points, HS)
		mean[0], cov[0] = HS.get_mean_cov()

		for iteration in range(num):
			image, hsv, dt, flow = params.get_frames()
			model.rz(-1,1,0.01)
			# model.tx(-1,1,10.)
			points = model.get_points()

			for i in range(len(points)):
				cv2.circle(image,(int(points[i][0]),int(points[i][1])),2,(0,0,255),-1)

			for i in range(len(points)-1):
				if i != 3 and i != 7 and i != 11:
					cv2.line(image,(int(points[i][0]),int(points[i][1])),(int(points[i+1][0]),int(points[i+1][1])),(255,0,0))
					if use_flow:
						cv2.line(image,(int(init_p[i][0]),int(init_p[i][1])),(int(init_p[i+1][0]),int(init_p[i+1][1])),(0,0,255))

			points = model.get_points(all_points=True)

			for i in range(len(points)):
				image[int(points[i][1]),int(points[i][0]), 1] = 255

			KH = util.descriptors(params, points, points, HS)
			mean[1], cov[1] = HS.get_mean_cov()

			# results[frame][iteration] = HS.HSIC(P_KH, KH)
			a = np.trace(np.linalg.inv(cov[1]) * cov[0])
			b = np.dot((mean[1] - mean[0]), np.dot(np.linalg.inv(cov[1]), (mean[1] - mean[0]).T))
			c = np.log(np.linalg.det(cov[1]) / np.linalg.det(cov[0]))

			results[frame][iteration] = 0.5 * (np.dot((mean[1] - mean[0]), np.dot(np.linalg.inv(cov[1]), (mean[1] - mean[0]).T)) + np.dot((mean[1] - mean[0]), np.dot(np.linalg.inv(cov[1]), (mean[1] - mean[0]).T)) - dims + np.log(np.linalg.det(cov[1]) / np.linalg.det(cov[0])))


			cv2.imshow("Neptune - Image", image)
			# cv2.imshow("Neptune - HSV",   hsv)
			# cv2.imshow("Neptune - DT",    dt)
			# cv2.imshow("Neptune - Flow",  flow[:,:,0])
			command = cv2.waitKey(1)

		model.rz(-1,1,-(num/2)*0.01)

		# params.update()
		# ----------- #

	for i in range(len(results)):
	 	plt.plot(results[i])

	plt.show()

def optimise(model, params, P_KH, N_KH, HS):
	state = np.random.rand(dims)*0.25 - 0.125
	optimisation = scipy.optimize.minimize(ARModel.flow_cost, state, 
		args=(model, params, P_KH, N_KH, HS, True), method='Nelder-Mead')
	print optimisation.x

def PSO(model, params, P_KH, N_KH, HS):

	print 'pso'

	state = [0. for i in range(dims)]

	P_KH = ARModel.get_KH(model, state, params, HS)
	params.update()

	for iteration in range(50):

		print iteration

		image, hsv, dt, flow = params.get_frames()

		state = ARModel.PSO(state, model, params, P_KH, N_KH, HS)

		P_KH = ARModel.get_KH(model, state, params, HS)

		# descriptors.append(ARModel.train(dims, model, state, motion=use_flow, update=True))

		# P_KH += descriptors[-1]
		params.update()

params = util.parameters()
image, hsv, dt, flow = params.get_frames()
model = ARModel.articulated_model(image.shape)

points = model.get_points(all_points=True)

HS = HSIC.HilbertSchmidt(len(points), 9)
HS_M = HSIC.HilbertSchmidt(len(points), 9)

dims = 15
use_flow = False

# --- TRAIN --- #
# ARModel.train(dims, model, [0. for i in range(dims)], motion=use_flow, update=False)
# import sys
# sys.exit()
P_KH, N_KH = ARModel.load_model()
# ------------- #

descriptors.append(np.copy(P_KH))

# import KL

# frames = [params.get_frames(), params.get_frames(10)]
# state = [[0. for i in range(dims)], np.random.rand(dims) * 10.]

# print '-->', KL.KL(model, frames, state, 9)

# test(model)
PSO(model, params, P_KH, N_KH, HS)
# optimise(model, params, P_KH, N_KH, HS)
import sys
sys.exit()

[params.update() for i in range(8)]

command = cv2.waitKey(1)
num = 40
results = results = [np.zeros(num) for i in range(len(descriptors))]

model.rz(-1,1,-(num/2)*0.01)
# model.tx(-1,1,-(num/2)*10.)

for iteration in range(num):
	image, hsv, dt, flow = params.get_frames()
	model.rz(-1,1,0.01)
	# model.tx(-1,1,10.)
	points = model.get_points()

	for i in range(len(points)):
		cv2.circle(image,(int(points[i][0]),int(points[i][1])),2,(0,0,255),-1)

	for i in range(len(points)-1):
		if i != 3 and i != 7 and i != 11:
			cv2.line(image,(int(points[i][0]),int(points[i][1])),(int(points[i+1][0]),int(points[i+1][1])),(255,0,0))
	
	points = model.get_points(all_points=True)

	for i in range(len(points)):
		image[int(points[i][1]),int(points[i][0]), 1] = 255
		# cv2.circle(image,(int(points[i][0]),int(points[i][1])),1,(0,0,255),)

	KH = util.descriptors(params, points, points, HS)

	for i in range(len(descriptors)):
		results[i][iteration] = HS.HSIC(descriptors[i], KH)

	cv2.imshow("Neptune - Image", image)

	__dir, __counter, __current_frame, __suffix = params.get_filename() 
	cv2.imwrite(__dir + str(__counter) + '__' + str(__current_frame) + __suffix, image)

	command = cv2.waitKey(1)
# ----------- #

for i in range(len(results)):
	plt.plot(results[i])

plt.show()







