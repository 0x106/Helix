import numpy as np
import cv2
import HSIC
import matplotlib.pyplot as plt

import activity_recognition as AR

def match_activity():

	files, labels = all_testing_files()
	file_lengths = get_data_lengths(files)

	print labels
	print file_lengths.T

	## ------------------------------ ##

	# length of the shortest video
	Q = np.min(file_lengths)

	print 'Q:', Q

	M = len(files)
	affinity_specific = np.zeros((M,M))
	affinity_general  = np.zeros((M,M))

	for i in range(M):
		for k in range(M):
			result, index, mean = AR.Neptune_with_temp_sync(files[i], files[k], _Q=Q)
			affinity_specific[i,k] = result
			affinity_general[i,k] = mean
			print i, k, affinity_specific[i,k], affinity_general[i,k]

	plt.subplot(121)
	AR.plot2d(affinity_specific)
	
	plt.subplot(122)
	AR.plot2d(affinity_general)
	plt.show()








def all_testing_files():
	files = []
	labels = []
	classes = []

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
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/01/01_0'+ str(2)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/01/01_0'+ str(3)+ '.amc')	
	labels.append('playground01')
	labels.append('playground01')

	# stylised walks 137
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/137/137_'+ str(12)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/137/137_'+ str(16)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/137/137_'+ str(20)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/137/137_'+ str(24)+ '.amc')		
	labels.append('walk137')
	labels.append('walk137')
	labels.append('walk137')
	labels.append('walk137')

	# sweeping 13
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/13/13_'+ str(23)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/13/13_'+ str(24)+ '.amc')	
	labels.append('sweeping13')
	labels.append('sweeping13')

	return files, labels

def get_data_lengths(files):
	data = AR.readData(files, -1, _PCA=False)
	file_lengths = np.zeros(len(data))
	for i in range(len(data)):
		file_lengths[i] = data[i].shape[1]
	return file_lengths

