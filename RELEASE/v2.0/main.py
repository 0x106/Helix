







import model as ARModel
import HSIC
import util
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

descriptors = []

def test():
	model = ARModel.wiggle(model, ((np.random.rand(dims) * 0.125) - 0.0625))

	command = cv2.waitKey(1)
	num = 400

	if use_flow:
		initial_model = ARModel.articulated_model(image.shape)
		initial_model.copy(model)
		initial_model_points = initial_model.get_points(all_points=True)
		init_p = initial_model.get_points()

		MPH = P_KH + util.descriptors(params, points, initial_model_points, HS_M, _train=True, _motion=True)
		results = results = [np.zeros(num) for i in range(6)]

	else:
		results = results = [np.zeros(num) for i in range(3)]

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
				if use_flow:
					cv2.line(image,(int(init_p[i][0]),int(init_p[i][1])),(int(init_p[i+1][0]),int(init_p[i+1][1])),(0,0,255))

		points = model.get_points(all_points=True)

		KH = util.descriptors(params, points, points, HS)

		results[0][iteration] = HS.HSIC(P_KH, KH)
		#results[1][iteration] =	HS.HSIC(N_KH, KH)


		results[2][iteration] = results[0][iteration]# - results[1][iteration]

		if use_flow:
			KH = util.descriptors(params, points, initial_model_points, HS_M, _motion=True)

			results[3][iteration] = HS_M.HSIC(MPH, KH)
			results[4][iteration] =	HS_M.HSIC(N_KH, KH)
			results[5][iteration] = results[3][iteration] - results[4][iteration]

		cv2.imshow("Neptune - Image", image)
		command = cv2.waitKey(1)
	# ----------- #

	for i in range(len(results)):
		plt.plot(results[i])

	plt.show()

def optimise(model, params, P_KH, N_KH, HS):
	state = np.random.rand(dims)*0.25 - 0.125
	optimisation = scipy.optimize.minimize(ARModel.compute_energy, state, 
		args=(model, params, P_KH, N_KH, HS, True), method='Nelder-Mead')
	print optimisation.x

def PSO(model, params, P_KH, N_KH, HS):
	state = [0. for i in range(dims)]

	for iteration in range(6):

		state = ARModel.PSO(state, model, params, P_KH, N_KH, HS)

		descriptors.append(ARModel.train(dims, model, state, motion=use_flow, update=True))

		P_KH += descriptors[-1]
		params.update()

params = util.parameters()
image, hsv, dt, flow = params.get_frames()
model = ARModel.articulated_model(image.shape)

points = model.get_points(all_points=True)

HS = HSIC.HilbertSchmidt(len(points), 7)
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

# test()
PSO(model, params, P_KH, N_KH, HS)
# optimise(model, params, P_KH, N_KH, HS)

[params.update() for i in range(4)]

command = cv2.waitKey(1)
num = 100
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







