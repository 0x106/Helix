
import cv2
import util
import obj
import activity_recognition
import articulated
import matplotlib.pyplot as plt
import HSIC
import numpy as np
import scipy.optimize

# obj.run()
# activity_recognition.run()
# activity_recognition.optical_flow_tracking()

def track_articulated():

	params = util.parameters()
	image, hsv, dt = params.get_frames()

	num_models = 1

	model = [articulated.articulated_model(image.shape) for i in range(num_models)]

	# model[1].copy(model[0])

	points = [model[i].get_points(all_points=True) for i in range(num_models)]
	HS = HSIC.HilbertSchmidt(len(points[0]))

	P_data, P_KH = util.descriptors(params, points[0], HS)

	init = np.random.rand(5)

	for i in range(50):
		params.update()
		articulated.PSO(init, model[0], params, P_KH, HS)

	# optimisation = scipy.optimize.minimize(articulated.optimise, init,
		# args=(model[0], params, P_KH, HS, True), method='Nelder-Mead', options={'xtol':1e-6, 'disp':False})

	# print optimisation

def test_articulated_tracking():
	params = util.parameters()
	image, hsv, dt = params.get_frames()

	num_models = 1

	model = [articulated.articulated_model(image.shape) for i in range(num_models)]

	# model[1].copy(model[0])

	points = [model[i].get_points(all_points=True) for i in range(num_models)]
	
	# init_model = articulated.articulated_model(image.shape, 100)
	# init_points = init_model.get_points(all_points=True)

	HS = HSIC.HilbertSchmidt(len(points[0]))
	# __P_data, __P_KH = util.descriptors(params, init_points, HS, True)	# init Cov
	P_data, P_KH = util.descriptors(params, points[0], HS, True)		# init KH

	print P_KH[:5,:5]

	results = [np.zeros(100) for i in range(num_models)]
	count = 0

	command = cv2.waitKey(1)

	for i in range(num_models):
		model[i].rz(-1,1,-50*0.01)

	while(command != 'q'):
		image, hsv, dt = params.get_frames()

		for i in range(num_models):
			model[i].rz(-1,1,0.01)

		points = [model[i].get_points() for i in range(num_models)]

		for k in range(num_models):
			for i in range(len(points[k])):
				cv2.circle(image,(int(points[k][i][0]),int(points[k][i][1])),2,(0,0,255))

			for i in range(len(points[k])-1):
				if i != 5:
					cv2.line(image,(int(points[k][i][0]),int(points[k][i][1])),(int(points[k][i+1][0]),int(points[k][i+1][1])),(255,0,0))

		points = [model[i].get_points(all_points=True) for i in range(num_models)]

		for k in range(num_models):
			for i in range(len(points[k])):
				cv2.circle(image,(int(points[k][i][0]),int(points[k][i][1])),1,(0,255,0))

		data, KH = [],[]
		for i in range(num_models):
			__data, __KH = util.descriptors(params, points[i], HS)
			data.append(__data)
			KH.append(__KH)

			results[i][count] = HS.__HSIC__(P_KH, data[i])

		print count, [results[i][count] for i in range(num_models)]

		cv2.imshow("Neptune - Image", image)

		if count == results[i].shape[0]-1:
			for i in range(num_models):
				plt.plot(results[i])
			plt.show()

		count += 1

		command = cv2.waitKey(1)






# ================= #
#    Main Control   #
# ================= #

A = np.zeros((3,3))
A[0,0] = 0.
A[1,1] = 0.
A[2,2] = 0.

A[0,1] = 10.
A[0,2] = 24.

A[1,2] = 8.

# activity_recognition.CHG_match()

# activity_recognition.plot_points(A,A)

activity_recognition.mocap_similarity()

# test_articulated_tracking()

# track_articulated()

# activity_recognition.run()
# activity_recognition.optical_flow_tracking()










































