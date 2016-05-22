import numpy as np
import cv2
import HSIC
import matplotlib.pyplot as plt

import activity_recognition as AR

def match_activity():

	files, labels = all_testing_files()

	print labels

	file_lengths = get_data_lengths(files)

	print file_lengths.T


























def all_testing_files():
	files = []
	labels = []

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

	# walking 16
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/16/16_'+ str(21)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/16/16_'+ str(22)+ '.amc')
	labels.append('walking16')
	labels.append('walking16')

	# playground 01
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/01/01_0'+ str(2)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/01/01_0'+ str(3)+ '.amc')	
	labels.append('playground01')
	labels.append('playground01')

	# walking 09
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/09/09_0'+ str(2)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/09/09_0'+ str(3)+ '.amc')	
	labels.append('walking09')
	labels.append('walking09')

	# running 08
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/08/08_0'+ str(2)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/08/08_0'+ str(3)+ '.amc')	
	labels.append('running08')
	labels.append('running08')

	# dancing 05
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/05/05_0'+ str(2)+ '.amc')
	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/05/05_0'+ str(3)+ '.amc')	
	labels.append('dancing05')
	labels.append('dancing05')

	# # WALKING
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/16/21/')
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/16/22/') 
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/16/32/') 
	# # files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/16/31/')
	# # files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/16/47/')

	# # PLAYGROUND
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/01/02/')  
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/01/03/')  
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/01/04/')  
	# # files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/01/06/')
	# # files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/01/10/')

	# # WALKING
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/09/02/')  
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/09/03/')  
	# files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/09/04/')  
	# # files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/09/05/')
	# # files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/09/06/')

 #    # RUNNING
 #    files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/08/02/')  
 #    files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/08/03/')  
 #    files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/08/04/') 
 #    # files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/08/08/')
 #    # files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/08/09/')

 #   	# DANCING
 #   	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/05/02/')  
 #   	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/05/03/')  
 #   	files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/05/04/')  
 #    # files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/05/18') 
 #    # files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/05/19') 


	return files, labels

def get_data_lengths(files):
	data = AR.readData(files, -1, _PCA=False)
	file_lengths = np.zeros(len(data))
	for i in range(len(data)):
		file_lengths[i] = data[i].shape[1]
	return file_lengths

