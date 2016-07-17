import numpy as np
import model as ARModel
import util
import cv2
import matplotlib.pyplot as plt

def hough(_image, _edges, name):

	# edges = cv2.Canny(gray,100,200,apertureSize = 3)
	img = cv2.cvtColor(np.copy(_image.astype(np.float32)), cv2.COLOR_BGR2GRAY).astype(np.uint8)
	# edges = cv2.Canny(img, 50, 150)

	lines = cv2.HoughLinesP(_edges, 0.1, 0.1, 6, None, 1, 10);

	print lines

	if lines != None:
		for line in lines:
			print line
			pt1 = (line[0,0],line[0,1])
			pt2 = (line[0,2],line[0,3])
			cv2.line(_image, pt1, pt2, (0,0,255), 2)
		cv2.imshow(name,_image)

def augment(_skeleton):

	h,w = _skeleton.shape[:2]
	points = []
	for r  in range(h):
		for c in range(w):
			if _skeleton[r,c] == 255:
				points.append([r,c])

	for i in range(len(points)):
		p1 = points[i]
		mindx = 1e12
		idx = i
		for k in range(len(points)):
			if i != k:
				p2 = points[k]
				d = np.sqrt(((p1[0]-p2[0])*(p1[0]-p2[0])) + ((p1[1]-p2[1])*(p1[1]-p2[1])))
				if d < mindx:
					mindx = d
					idx = k
		cv2.line(_skeleton, (points[i][1], points[i][0]), (points[idx][1], points[idx][0]), 255)
	return _skeleton

def skeleton(_image):

	img = cv2.cvtColor(np.copy(_image.astype(np.float32)), cv2.COLOR_BGR2GRAY).astype(np.uint8)

	img = cv2.blur(img, (19,19))

	skel = np.zeros(img.shape,np.uint8)
	edges = cv2.Canny(img, 50, 150)
	size = np.size(img)
	ret,img = cv2.threshold(img,160,255,0)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done = False

	while( not done):
		eroded = cv2.erode(img,element)
		temp = cv2.dilate(eroded,element)
		temp = cv2.subtract(img,temp)
		skel = cv2.bitwise_or(skel,temp)
		img = eroded.copy()

		zeros = size - cv2.countNonZero(img)
		if zeros==size:
			done = True

	# hough(_image, np.copy(skel), 'dst')
	# skel = augment(skel)
	# joint = cv2.bitwise_or(skel, edges)

	cv2.imshow("Skeleton",skel)
	# cv2.imshow("Edges",edges)
	# cv2.imshow("Joint1",joint)
	cv2.waitKey(0)

params = util.parameters()

for i in range(100):
	image, hsv, dt, flow = params.get_frames(1)
	skeleton(image)
