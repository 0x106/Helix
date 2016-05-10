
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

params = util.parameters()
image, hsv, dt = params.get_frames()

num_models = 2

model = [articulated.articulated_model(image.shape) for i in range(num_models)]

model[1].copy(model[0])

## draw the model for testing
# points = model.get_points()
# for i in range(len(points)):
# 	cv2.circle(image,(int(points[i][0]),int(points[i][1])),2,(0,0,255))
# for i in range(len(points)-1):
# 	if i != 6:
# 		cv2.line(image,(int(points[i][0]),int(points[i][1])),(int(points[i+1][0]),int(points[i+1][1])),(255,0,0))
# points = model.get_points(all_points=True)

# for i in range(len(points)):
#  	cv2.circle(image,(int(points[i][0]),int(points[i][1])),1,(0,255,0))

# cv2.imshow("Neptune - Image", image)
# cv2.waitKey(0)

points = [model[i].get_points(all_points=True) for i in range(num_models)]
HS = HSIC.HilbertSchmidt(len(points[0]))

P_data, P_KH = util.descriptors(params, points[0], HS)

# temp_models = [articulated.articulated_model(image.shape) for i in range(num_models*10)]

# for i in range(num_models*10):
# 	temp_points = temp_models[i].get_points(all_points=True)
# 	__P_data, __P_KH = util.descriptors(params, temp_points, HS)
# 	P_KH += __P_KH

P_KH /= (num_models*10)

# init = np.random.rand(3)

# articulated.PSO(init, params, P_KH, HS)

# optimisation = scipy.optimize.minimize(articulated.optimise, init,
	# args=(params, P_KH, HS), method='Nelder-Mead', options={'xtol':1e-6, 'disp':False})

# print optimisation

results = [np.zeros(100) for i in range(num_models)]
count = 0

command = cv2.waitKey(1)

for i in range(num_models):
	model[i].rz(-1,1,-50*0.01)

while(command != 'q'):

	# params.update()
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
# 	# # cv2.imshow("Neptune - HSV", hsv)
# 	# # cv2.imshow("Neptune - DT", dt)

	if count == results[i].shape[0]-1:
		for i in range(num_models):
			plt.plot(results[i])
		plt.show()

	count += 1

	command = cv2.waitKey(1)

# image, hsv, dt = params.get_frames()

# # points = model.get_points()

# # for i in range(len(points)-1):
# # 	if i != 4:
# # 		cv2.line(image,(int(points[i][0]),int(points[i][1])),(int(points[i+1][0]),int(points[i+1][1])),(255,0,0))
# # for i in range(len(points)):
# # 	cv2.circle(image,(int(points[i][0]),int(points[i][1])),2,(0,0,255))

# points = model.get_points(all_points=True)

# for i in range(len(points)):
# 	print '-->', points[i]
# 	cv2.circle(image,(int(points[i][0]),int(points[i][1])),1,(0,255,0))

# cv2.imshow("Neptune - HSV", hsv)
# cv2.imshow("Neptune - DT", dt)
# cv2.imshow("Neptune - Image", image)
# cv2.waitKey(0)


























































