import numpy as np
import cv2
import HSIC
import matplotlib.pyplot as plt

import activity_recognition as AR

import sklearn
from sklearn.cluster import KMeans

# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


def temp_sync():

	files = []
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/01/01_0'+ str(2)+ '.amc')
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/86/86_'+ str(15)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/86/86_0'+ str(9)+ '.amc')
	data, PCA_DIMS1 = AR.readData(files, -1, _PCA=True)

	ref_files = []
	ref_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/105/105_'+ str(34)+ '.amc')
	ref_data, PCA_DIMS2 = AR.readData(ref_files, -1, _PCA=True)

	PCA_DIMS = max(PCA_DIMS1, PCA_DIMS2)

	print ref_data[0].shape, PCA_DIMS

	M, N = len(files), 200

	# plt.subplot(121)
	# plt.plot(ref_data[0][0,:])
	# plt.plot(ref_data[0][1,:])
	# plt.plot(ref_data[0][2,:])

	# plt.subplot(122)
	# plt.plot(ref_data[0][:PCA_DIMS*2,0])

	# plt.show()

	HS = HSIC.HilbertSchmidt(N)

	results = []

	Q = 720
	offset = 420

	print data[0].shape[1]-Q-10

	for iter in range(0,data[0].shape[1]-Q-10, 30):

		cov_data = np.copy(data[0][:PCA_DIMS,iter:iter+Q])

		for r in range(4):
			cov_data = np.append(cov_data, np.copy(ref_data[0][:PCA_DIMS, offset + int(10.*r) : offset + int(10.*r) + Q]), 1)

		HS.set_covariance(cov_data)
		KH = np.zeros((N,N))

		for r in range(4):
			KH += HS.AR_get_KH(AR.downsample(ref_data[0][:PCA_DIMS, offset + int(10.*r) : offset + int(10.*r) + Q], N))

		Y = AR.downsample(data[0][:PCA_DIMS,iter:iter+Q],N)
		results.append(HS.__HSIC__(KH, Y))
		print iter, results[-1]

	plt.plot(results)
	plt.show()


def match_activity():

	files, labels = all_testing_files()
	file_lengths = get_data_lengths(files)

	print labels
	print file_lengths.T

	## ------------------------------ ##

	# length of the shortest video
	Q = np.min(file_lengths)

	print 'Q:', Q

	M, N, PCA_DIMS = len(files), 200, 8
	affinity_specific = np.zeros((M,M))
	affinity_general  = np.zeros((M,M))

	data = AR.readData(files, -1, _PCA=True)

	## ---------- set covariance ---------- ##
	HS = HSIC.HilbertSchmidt(N)
	covariance_data = np.copy(data[0][:PCA_DIMS, :])
	for i in range(1,M):
		covariance_data = np.append(covariance_data, data[i][:PCA_DIMS, :], 1)
	HS.set_covariance(covariance_data)
    ## ------------------------------------ ##

	for i in range(M):
		for k in range(i, M):
			result, index, mean = AR.Neptune_with_temp_sync(data[i], data[k],
						HS, N, PCA_DIMS, _Q=Q)
			affinity_specific[i,k] = result
			affinity_general[i,k] = mean

			affinity_specific[k,i] = result
			affinity_general[k,i] = mean

			print i, k, affinity_specific[i,k], affinity_general[i,k]

	clusters = sklearn.cluster.spectral_clustering(affinity_general, n_clusters=4)
	print clusters

	plt.subplot(121)
	AR.plot2d(affinity_specific)
	
	plt.subplot(122)
	AR.plot2d(affinity_general)
	plt.show()

def all_testing_files():
	files = []
	labels = []
	classes = []

	# rolling 128
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/128/128_'+ str(10)+ '.amc')
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/128/128_'+ str(11)+ '.amc')	
	# labels.append('rolling128')
	# labels.append('rolling128')

	# backstroke 126
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/126/126_0'+ str(1)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/126/126_0'+ str(2)+ '.amc')
	labels.append('backstroke126')
	labels.append('backstroke126')	
	
	# freestyle 126
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/126/126_'+ str(10)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/126/126_'+ str(11)+ '.amc')
	labels.append('freestyle126')
	labels.append('freestyle126')

	# playground 01
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/01/01_0'+ str(2)+ '.amc')
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/01/01_0'+ str(3)+ '.amc')	
	# labels.append('playground01')
	# labels.append('playground01')q

	# walks 136
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/136/136_'+ str(21)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/136/136_'+ str(22)+ '.amc')
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/137/137_'+ str(20)+ '.amc')
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/137/137_'+ str(24)+ '.amc')		
	labels.append('walk136')
	labels.append('walk136')
	# labels.append('walk137')
	# labels.append('walk137')

	# getting up 140
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/140/140_0'+ str(3)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/140/140_0'+ str(4)+ '.amc')	
	labels.append('gettingup140')
	labels.append('gettingup140')

	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/140/140_0'+ str(2)+ '.amc')
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/140/140_0'+ str(8)+ '.amc')	
	# labels.append('gettingup140')
	# labels.append('gettingup140')

	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/90/90_'+ str(30)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/90/90_'+ str(31)+ '.amc')	
	labels.append('russiandance90')
	labels.append('russiandance90')

	# sweeping 13
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/13/13_'+ str(23)+ '.amc')
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/13/13_'+ str(24)+ '.amc')	
	# labels.append('sweeping13')
	# labels.append('sweeping13')

	return files, labels

def get_data_lengths(files):
	data = AR.readData(files, -1, _PCA=False)
	file_lengths = np.zeros(len(data))
	for i in range(len(data)):
		file_lengths[i] = data[i].shape[1]
	return file_lengths

def rbf(x,y,s):
	z = x-y
	q = (z*z)/(2.*s)
	return np.exp(-q)

def magnitude_vs_frequency(): 

	N = 100
	X = np.linspace(0, 6.*np.pi, num=N)

	Y = 16.*np.sin(X)

	H = np.zeros((N,N))
	K = np.zeros((N,N))
	L = np.zeros((N,N))

	results = np.zeros(40)
	results_norm = np.zeros(40)
	mag_results, freq_results, results3d = [], [], []

	for i in range(N):
		for k in range(N):
			H[i,k] = 0. - (1./N)
		H[i,i] = 1. - (1./N)

	# data = Y
	# for mag in range(1,41):
	# 	for freq in range(1,41):
	# 		Y__ = mag * np.sin((freq*0.01) * X)
	# 		data = np.append(data, Y__)
	# var = np.var(data)

	# print 'variance:', var

	# for mag in range(1,41):
	for freq in range(1,41):

		Y__ = freq * np.sin( X)

		data = Y
		data = np.append(data, Y__)
		var = np.var(data)

		for i in range(N):
			for k in range(N):
				K[i,k] = rbf(Y[i], Y[k], var)
				L[i,k] = rbf(Y__[i], Y__[k], var)

		KH = np.dot(K,H)
		LH = np.dot(L,H)

		# results[j-1] = ((1. / (N*N)) * np.trace(np.dot(KH,LH)))
		results_norm[freq-1] = ((1. / (N*N)) * np.trace(np.dot(KH,LH))) / np.sqrt(((1. / (N*N)) * np.trace(np.dot(KH,KH)) )*((1. / (N*N)) * np.trace(np.dot(LH,LH)) ))

		# mag_results.append(mag)
		# freq_results.append(freq*0.01)
		# results3d.append(((1. / (N*N)) * np.trace(np.dot(KH,LH))) / np.sqrt(((1. / (N*N)) * np.trace(np.dot(KH,KH)) )*((1. / (N*N)) * np.trace(np.dot(LH,LH)) )))

		print  freq*0.01, results_norm[freq-1]#results[j-1], results_norm[j-1]


	# ax.scatter(mag_results, freq_results, results3d, c=results3d)
	# plt.show()
	# plt.subplot(121)
	# ax.scatter(results_norm)
	# plt.subplot(122)
	plt.plot(results_norm)
	plt.show()
