# fig = plt.figure()

# position = [100,100]
# init = Falsed


# def plot2d(input):
#     im = plt.imshow(input, interpolation='nearest', cmap='spring')
#     plt.gca().invert_yaxis()
#     # plt.colorbar()

# def test():
# 	x = np.linspace(-10, 10, num=100)
# 	y = np.exp(-(x*x/20.))

# 	plt.plot(y)
# 	plt.show()

# # def draw_rect(event,x,y,flags,param):

# # 	global rect, init

# # 	position[0] = y
# # 	position[1] = x

# # 	if event == cv2.EVENT_LBUTTONDOWN:
# # 		init = True

# # def playback(dir):

# # 	command = cv2.waitKey(1)
# # 	frame = 1
# # 	while(command != 'q'):

# # 		file = dir + str(frame) + '.png'

# # 		image = cv2.imread(file, cv2.IMREAD_COLOR)

# # 		grey = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2GRAY)
# # 		edges = cv2.Canny(np.copy(grey), 50,150)

# # 		edges = cv2.bitwise_not(np.copy(edges))

# # 		dt = cv2.distanceTransform(np.copy(edges), cv2.DIST_L2, maskSize=5)

# # 		cv2.normalize(np.copy(dt), dt, alpha=0., beta=100.0, norm_type=cv2.NORM_MINMAX)

# # 		hsv = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2HSV)

# # 		cv2.imshow("Edges", edges)
# # 		cv2.imshow("Neptune", image)
# # 		cv2.imshow("DT", dt)
# # 		cv2.imshow("hsv", hsv)
# # 		command = cv2.waitKey(1)
# # 		frame += 1
# def get_descriptors(image, state, features, N, K):

# 	descriptor = np.zeros((K,N))

# 	pose = ((state[2] * np.copy(features)) + np.copy(state[:2])).astype(int)

# 	grey = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2GRAY)
# 	edges = cv2.Canny(np.copy(grey), 50,150)

# 	edges = cv2.bitwise_not(np.copy(edges))

# 	dt = cv2.distanceTransform(np.copy(edges), cv2.DIST_L2, maskSize=5)

# 	cv2.normalize(np.copy(dt), dt, alpha=0., beta=100.0, norm_type=cv2.NORM_MINMAX)

# 	hsv = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2HSV)

# 	SAMPLE_HSV = True

# 	for i in range(N):

# 		# if SAMPLE_HSV:
# 		descriptor[0,i] = hsv[pose[i,0],pose[i,1],0]
# 		descriptor[1,i] = hsv[pose[i,0],pose[i,1],1]
# 		descriptor[2,i] = hsv[pose[i,0],pose[i,1],2]
# 		# else:
# 		descriptor[3,i] = image[pose[i,0],pose[i,1],0]
# 		descriptor[4,i] = image[pose[i,0],pose[i,1],1]
# 		descriptor[5,i] = image[pose[i,0],pose[i,1],2]

# 		descriptor[3,i] = dt[pose[i,0],pose[i,1]]

# 		# cv2.circle(image,(int(pose[i,1]),int(pose[i,0])),1,(255,0,0), -1)

# 	return descriptor

# def rbf(x, y, S):
# 	z = x-y
# 	q = np.dot(z.T, np.dot(S,z))
# 	return np.exp(-q)

# 	# z = x-y
# 	# q = np.dot(x.T, np.dot(S,z))
# 	# print q, np.exp(-1.*(q))
# 	# return np.exp(-q)

# def HSIC(KH, Y, H, S):

# 	N = X.shape[1]
	
# 	K,L,LH = np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))
# 	# K,L, KH, LH = np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))

# 	for i in range(N):
# 		for k in range(N):
# 			# K[i,k] = rbf(X[:,i], X[:,k], S)
# 			L[i,k] = rbf(Y[:,i], Y[:,k], S)

# 	# KH = np.dot(K,H)
# 	LH = np.dot(L,H)

# 	return ((1. / (N*N)) * np.trace(np.dot(KH,LH))) / (np.sqrt((1. / (N*N)) * np.trace(np.dot(LH,LH))))

# def main():

# 	global position, init

# 	print 'Running python Neptune'

# 	dir = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/ball/ball'
	
# 	# playback(dir)

# 	file = dir + str(1) + '.png'
# 	image = cv2.imread(file, cv2.IMREAD_COLOR)

# 	cv2.namedWindow('Neptune')
# 	cv2.setMouseCallback('Neptune',draw_rect)

# 	command = cv2.waitKey(1)

# 	state = [-1,-1, 1.0]
# 	P = 2
# 	negative = []
# 	for i in range(P):
# 		negative.append([-1,-1,1.0])
# 	radius = 46

# 	while(command != 'q'):

# 		file = dir + str(1) + '.png'
# 		image = cv2.imread(file, cv2.IMREAD_COLOR)

# 		cv2.circle(image,(int(position[1]),int(position[0])),radius,(0,0,255))

# 		cv2.circle(image,(int(state[1]),int(state[0])),radius,(255,0,0))

# 		for i in range(P):
# 			cv2.circle(image,(int(negative[i][1]),int(negative[i][0])),radius,(0,255,0))

# 		if init:
# 			if state[0] == -1:
# 				state[:2] = position
# 			else:
# 				num = 0
# 				for i in range(P):
# 					if negative[i][0] == -1:
# 						negative[i][:2] = position
# 						num += 1
# 						break
# 				if num == 0:
# 					break
# 			init = False

# 		cv2.imshow("Neptune", image)
# 		command = cv2.waitKey(1)

# 	N,K = 100,7		# 10 features, 3(4) dimensions per feature --> r,g,b,(d)
# 	features = init_features(N, radius)
# 	Y = get_descriptors(image, state, features, N, K)
# 	Z = []	
# 	for i in range(P):
# 		Z.append(get_descriptors(image, negative[i], features, N, K))

# 	# cv2.imshow("Neptune", image)
# 	# cv2.waitKey(0)

# 	H = np.zeros((N,N))
# 	# S = np.eye(K) * (1./((K*12000)))
# 	S = np.eye(K) * (1./((K*10000)))
# 	S[3,3] = (1./((K*2000)))
# 	for i in range(N):
# 		for k in range(N):
# 			H[i,k] = 0. - (1./N)
# 		H[i,i] = 1. - (1./N)

# 	pose = np.copy(state)

# 	# results = np.zeros((20,20))
# 	xdx,ydx, results = [],[],[]

# 	image = cv2.imread(file, cv2.IMREAD_COLOR)

# 	negs = []
# 	K = np.zeros((N,N))
# 	for i in range(N):
# 		for k in range(N):
# 			K[i,k] = rbf(Y[:,i], Y[:,k], S)

# 	positive = np.dot(K,H)
# 	negs = np.zeros((N,N))
# 	for j in range(P):
# 		for i in range(N):
# 			for k in range(N):
# 				K[i,k] = rbf(Z[j][:,i], Z[j][:,k], S)

# 		negs += np.dot(K,H)

# 	negs /= float(P)

# 	for row in range(-10,10):

# 		# xdx = 0
# 		for col in range(-10,10):

# 			# image = cv2.imread(file, cv2.IMREAD_COLOR)

# 			pose[:2] = row, col
# 			pose[:2] += state[:2]

# 			X = get_descriptors(image, pose, features, N, K)

# 			# Y is fixed, X changes
# 			# print row, col, HSIC(Y, X, H, S)

# 			xdx.append(col)

# 			# results[ydx,xdx] = HSIC(Y, X, H, S)

# 			results.append(HSIC(positive, X, H, S) - HSIC(negs,X,H,S))
# 			print row, col, results[-1]
# 			# print ydx, xdx, results[ydx,xdx]

# 			# xdx += 1
# 		# ydx += 1
# 			ydx.append(row)

# 			# cv2.imshow("Neptune", image)
# 			# cv2.waitKey(/1)

# 	ax = fig.add_subplot(111, projection='3d')
# 	ax.scatter(xdx, ydx, results, c=results)

# 	# plot2d(results)
# 	plt.show()

# 	# for i in range(100):

# 	# 	image = cv2.imread(file, cv2.IMREAD_COLOR)

# 	# 	pose[:2] = ((np.random.rand(2) * 20.0) - 10.0) + state[:2]

# 	# 	if pose[0] > (image.shape[0] - (radius * pose[2])):
# 	# 		pose[0] = (image.shape[0] - (radius * pose[2]))
# 	# 	if pose[1] > (image.shape[1] - (radius * pose[2])):
# 	# 		pose[1] = (image.shape[1] - (radius * pose[2]))

# 	# 	X = get_descriptors(image, pose, features, N, K)

# 	# 	# Y is fixed, X changes
# 	# 	print pose[:2], HSIC(Y, X, H, S)

# 	# 	# print '---------------------------------------------------'

# 	# 	cv2.imshow("Neptune", image)
# 	# 	cv2.waitKey(1)


# # test()
# initialise()
# main()