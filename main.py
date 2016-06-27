# 
import cv2
import util
import obj
import activity_recognition
import articulated
import matplotlib.pyplot as plt
import HSIC
import numpy as np
import scipy.optimize

import full_testing

model_p_file = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/dev/model_positive.npy'
model_n_file = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/dev/model_negative.npy'

# obj.run()
# activity_recognition.run()
# activity_recognition.optical_flow_tracking()

def load_model():
	return np.load(model_p_file), np.load(model_n_file)

def train(dims):
	params = util.parameters()
	image, hsv, dt = params.get_frames()

	model = articulated.articulated_model(image.shape)

	points = model.get_points(all_points=True)
	
	HS = HSIC.HilbertSchmidt(len(points))

	state = [0. for i in range(dims)]
	P_KH = get_KH(model, state, params, HS)

	state[2] = -40 * 0.01
	N_KH = get_KH(model, state, params, HS)

	print 'Training model...'

	# --------- train --------- #
	k = 2
	for i in range(-80, 80):
		if i < -14 or i > 14:
			state = [0. for idx in range(dims)]
			state[2] = i * 0.005
			N_KH += get_KH(model, state, params, HS)
		else:#if i >= -6 or i <= 6:
			for k in range(dims):
				state = [0. for idx in range(dims)]
				state[k] = i * 0.005
				P_KH += get_KH(model, state, params, HS)
	# ------------------------------------- #

	np.save(model_p_file, P_KH)
	np.save(model_n_file, N_KH)

def track():
	params = util.parameters()
	image, hsv, dt = params.get_frames()

	model = articulated.articulated_model(image.shape)

	points = model.get_points(all_points=True)
	
	HS = HSIC.HilbertSchmidt(len(points))

	dims = 9

	state = [0. for i in range(dims)]
	P_KH = get_KH(model, state, params, HS)

	state[2] = -40 * 0.01
	N_KH = get_KH(model, state, params, HS)

	print 'Training model...'

	# --------- train --------- #
	k = 2
	for i in range(-40, 40,2):
		if i >= -5 or i <= 5:
			for k in range(2, dims):
				state = [0. for idx in range(dims)]
				state[k] = i * 0.01
				P_KH += get_KH(model, state, params, HS)
	# ------------------------------------- #

	state = [0.0 for i in range(dims)]

	for i in range(10):
		params.update()

	for i in range(50):
		params.update()

		print 'Initial state:', state

		optimisation = scipy.optimize.minimize(articulated.compute_energy, state, 
			args=(model, params, P_KH, N_KH, HS, True), method='Nelder-Mead')

		print 'output:', optimisation.x

		state = optimisation.x
		state += [((np.random.rand() * 0.1) - 0.05) for i in range(dims)]

def get_KH(model, state, params, HS):
	model = articulated.wiggle(model, state)
	points = model.get_points(all_points=True)
	data, KH = util.descriptors(params, points, HS, True)		
	model = articulated.wiggle(model, state, inverse=True)
	return KH

def track_articulated():

	params = util.parameters()
	image, hsv, dt = params.get_frames()

	points = model.get_points(all_points=True)

	print len(points)
	
	HS = HSIC.HilbertSchmidt(len(points))
	N_HS = HSIC.HilbertSchmidt(len(points))

	P_data, P_KH = util.descriptors(params, points, HS, True)		# init KH

	N_KH = get_KH(model, [0.,0.,-40*0.01, 0.,0.], params, HS)

	# --------- positive examples --------- #

	ignore = [0]
	for i in range(-4, 4):
		if i not in ignore:
			N_KH += get_KH(model, [0.,0.,i*0.1,0.,0.], params, HS)

	# --------- negative examples --------- #

	ignore = [-40, -20, -10, 0, 10, 20]
	for i in range(-40, 40, 10):
		if i not in ignore:
			N_KH += get_KH(model, [0.,0.,i*0.1,0.,0.0], params, HS)

	N_KH += get_KH(model, [0.,0.,0.,-0.1,0.], params, HS)
	N_KH += get_KH(model, [0.,0.,0.,0.1,0.], params, HS)
	N_KH += get_KH(model, [0.,0.,0.,0.,-0.1], params, HS)
	N_KH += get_KH(model, [0.,0.,0.,0.,0.1], params, HS)

	# ------------------------------------- #

	state = np.random.rand(5)

	for i in range(50):
		params.update()

		state = articulated.PSO(state, model, params, P_KH, N_KH, HS)

		model = articulated.wiggle(model, state)

		P_KH += get_KH(model, [0.,0.,0.,0.,0.], params, HS)
		
		N_KH += get_KH(model, [0.,0.,state[2] - 0.1,0.,0.], params, HS)
		N_KH += get_KH(model, [0.,0.,state[2] + 0.1,0.,0.], params, HS)

		N_KH += get_KH(model, [0.,0.,state[3] - 0.1,0.,0.], params, HS)
		N_KH += get_KH(model, [0.,0.,state[3] + 0.1,0.,0.], params, HS)

		N_KH += get_KH(model, [0.,0.,state[4] - 0.1,0.,0.], params, HS)
		N_KH += get_KH(model, [0.,0.,state[4] + 0.1,0.,0.], params, HS)



	# optimisation = scipy.optimize.minimize(articulated.optimise, state,
		# args=(model[0], params, P_KH, HS, True), method='Nelder-Mead', options={'xtol':1e-6, 'disp':False})

	# print optimisation

def test_articulated_tracking():
	params = util.parameters()
	image, hsv, dt = params.get_frames()

	model = articulated.articulated_model(image.shape)

	points = model.get_points(all_points=True)
	
	HS = HSIC.HilbertSchmidt(len(points))

	dims = 13

	# train(dims)
	# return

	if False:
		state = [0. for i in range(dims)]
		P_KH = get_KH(model, state, params, HS)

		state[2] = -40 * 0.01
		N_KH = get_KH(model, state, params, HS)

		print 'Training model...'

		# --------- train --------- #
		k = 2
		for i in range(-40, 40,2):
			if i < -6 or i > 6:
				state = [0. for idx in range(dims)]
				state[2] = i * 0.01
				N_KH += get_KH(model, state, params, HS)
			else:#if i >= -6 or i <= 6:
				for k in range(dims):
					state = [0. for idx in range(dims)]
					state[k] = i * 0.01
					P_KH += get_KH(model, state, params, HS)
		# ------------------------------------- #
	else:
		P_KH, N_KH = load_model()

	results = [np.zeros(200) for i in range(5)]
	count = 0

	# points = model.get_points()

	# for i in range(len(points)):
	# 	cv2.circle(image,(int(points[i][0]),int(points[i][1])),2,(0,0,255),-1)

	# for i in range(len(points)-1):
	# 	if i != 3 and i != 7:
	# 		cv2.line(image,(int(points[i][0]),int(points[i][1])),(int(points[i+1][0]),int(points[i+1][1])),(255,0,0))

	# cv2.imshow("Neptune - Image", image)

	# command = cv2.waitKey(0)

	# return

	state = [((np.random.rand() * 0.2) - 0.1) for i in range(dims)]
	# # state = [0. for i in range(dims)]
	state[0] *= 100.
	state[1] *= 100.

	# state = np.random.rand() * 0.2 - 0.1

	print 'Initial state:', state

	for i in range(50):
		params.update()

		state = articulated.PSO(state, model, params, P_KH, N_KH, HS)

		for k in range(2,9):

			P_KH += get_KH(model, [0. for i in range(dims)], params, HS)

			update = [0. for i in range(dims)]
			update[k] = 0.01
			P_KH += get_KH(model, update, params, HS)

			update = [0. for i in range(dims)]
			update[k] = -0.01
			P_KH += get_KH(model, update, params, HS)

		update = [0. for i in range(dims)]
		update[k] = 0.1
		N_KH += get_KH(model, update, params, HS)

		update = [0. for i in range(dims)]
		update[k] = -0.1
		N_KH += get_KH(model, update, params, HS)

		
	# optimisation = scipy.optimize.minimize(articulated.compute_energy, state, 
		# args=(model, params, P_KH, N_KH, HS, True), method='Nelder-Mead')
# 
	# print optimisation.x

	# return

	command = cv2.waitKey(1)

	# params.update()

	# model.rz(1,3,-50*0.01)
	model.rz(-1,1,-100*0.01)
	while(command != 'q'):
		# params.apply_noise()
		image, hsv, dt = params.get_frames()

		model.rz(-1,1,0.01)
		# model.rz(1,3,0.01)

		points = model.get_points()

		for i in range(len(points)):
			cv2.circle(image,(int(points[i][0]),int(points[i][1])),2,(0,0,255),-1)

		for i in range(len(points)-1):
			if i != 3:
				cv2.line(image,(int(points[i][0]),int(points[i][1])),(int(points[i+1][0]),int(points[i+1][1])),(255,0,0))

		points = model.get_points(all_points=True)

		for i in range(len(points)):
			cv2.circle(image,(int(points[i][0]),int(points[i][1])),1,(0,255,0),-1)

		__data, __KH = util.descriptors(params, points, HS, True)

		results[0][count] = HS.__HS_IC__(P_KH, __KH)
		results[1][count] =	HS.__HS_IC__(N_KH, __KH)
		results[2][count] = results[0][count] - results[1][count]

		print count, [results[i][count] for i in range(3)]

		cv2.imshow("Neptune - Image", image)

		# cv2.imwrite('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CHG/pose_results/'+str(count)+'.png', image)


		if count == results[i].shape[0]-1:
			plt.plot(results[0])
			plt.plot(results[1])
			plt.plot(results[2])
			plt.show()
			return

		count += 1

		command = cv2.waitKey(1)


# ================= #
#    Main Control   #
# ================= #

# A = np.random.rand(100,100)
# B = np.random.rand(100,100)

# for i in range(1000):
# 	res = np.trace(np.dot(A,B))
# 	# res = np.dot(np.ravel(A), np.ravel(B))

# track_articulated()
test_articulated_tracking()
# track()

# full_testing.temp_sync()
# full_testing.audio_video()
# full_testing.temp_sync_mocap_image()
# full_testing.magnitude_vs_frequency()

# full_testing.match_activity()

#vactivity_recognition.Neptune_cluster_TS()

# import os, sys

# path = "/Users/jordancampbell/Desktop/Helix/code/pyNeptune/dev/misc"

# os.mkdir(path) ;

# A = np.zeros((3,3))
# A[0,0] = 0.
# A[1,1] = 0.
# A[2,2] = 0.

# A[0,1] = 10.
# A[0,2] = 24.

# A[1,2] = 8.

# activity_recognition.KTH()
# 
# activity_recognition.Neptune_cluster_KH()
# activity_recognition.Neptune_cluster()

# activity_recognition.CHG_match()

# activity_recognition.plot_points(A,A)

# activity_recognition.match_sample()

# num_files = activity_recognition.mocap_similarity()
# activity_recognition.match_sample_image(num_files)

# test_articulated_tracking()

# track_articulated()

# activity_recognition.run()
# activity_recognition.optical_flow_tracking()











































# 