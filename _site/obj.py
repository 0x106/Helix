# object recognition


import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import util
import HSIC


# def run():

# 	# file = util.dir + str(util.current_frame) + '.png'
# 	image = cv2.imread(util.file, cv2.IMREAD_COLOR)
# 	display = cv2.imread(util.file, cv2.IMREAD_COLOR)

# 	P, N = util.initialise(image)

# 	x, y, r = [],[],[]

# 	pose = util.model()
# 	pose.set(P[0].state, image, util.features[0])

# 	delta = np.zeros(3)

# 	x, y, rp, rn, rj = [],[],[], [], []

# 	P_KH = np.zeros((HSIC.N, HSIC.N))
# 	N_KH = np.zeros((HSIC.N, HSIC.N))

# 	for i in range(len(P)):
# 		P_KH += P[i].KH
# 		N_KH += N[i].KH

# 	# P_KH /= float(len(P))
# 	# N_KH /= float(len(P))

# 	init_pose = np.zeros(3)
# 	init_pose[:] = P[0].state + ((np.random.rand(3) * 10.) - 5.)
# 	init_pose[2] = 1.0

# 	for i in range(100):

# 		image = cv2.imread(util.file, cv2.IMREAD_COLOR)

# 		print 'Initial pose:', init_pose

# 		optimisation = scipy.optimize.minimize(optimise, init_pose, args=(pose, image, P_KH), method='Nelder-Mead', options={'xtol':1e-6, 'disp':False})

# 		pose.show_state(util.features[0])

# 		P_KH  += pose.KH

# 		print 'Final state:', optimisation.x

# 		print '-----------------------------------\n'

# 		util.current_frame += 2
# 		util.file = util.dir + str(util.current_frame) + '.png'
# 		for i in range(len(optimisation.x)):
# 			init_pose[i] = optimisation.x[i]

# 	# util.current_frame = 169
# 	# util.file = util.dir + str(util.current_frame) + '.png'
# 	# image = cv2.imread(util.file, cv2.IMREAD_COLOR)

# 	# for row in range(326-20,326+20, 4):
# 	# 	for col in range(424-20,424+20, 4):
			
# 	# 		delta[:] = row, col, 0.95

# 	# 		pose.set(delta, image, util.features[0])

# 	# 		pose.show_state(util.features[0])
			
# 	# 		x.append(col)
# 	# 		y.append(row)
# 	# 		# r.append(HSIC.HSIC(P[0].data, pose.data, HSIC.H, HSIC.S))
# 	# 		rp.append(HSIC.__HSIC__(P_KH, pose.data, HSIC.H, HSIC.S))
# 	# 		rn.append(HSIC.__HSIC__(N_KH, pose.data, HSIC.H, HSIC.S))
# 	# 		rj.append(rp[-1] - rn[-1])

# 	# 		print y[-1], x[-1], rp[-1], rn[-1], rj[-1]

# 	# fig = plt.figure()
# 	# ax = fig.add_subplot(131, projection='3d')
# 	# ax2 = fig.add_subplot(132, projection='3d')
# 	# ax3 = fig.add_subplot(133, projection='3d')
# 	# ax.scatter(x, y, rp, c=rp)
# 	# ax2.scatter(x, y, rn, c=rn)
# 	# ax3.scatter(x, y, rj, c=rj)
# 	# plt.show()



# def optimise(_delta, _pose, _image, _P_KH):
	
# 	_pose.set(_delta, _image, util.features[0])

# 	# _pose.show_state(util.features[0])

# 	return -1. * HSIC.__HSIC__(_P_KH, _pose.data, HSIC.H, HSIC.S)





