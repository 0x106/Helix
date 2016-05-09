
import cv2
import util
import obj
import activity_recognition
import articulated

# obj.run()
# activity_recognition.run()
# activity_recognition.optical_flow_tracking()

params = util.parameters()
image, hsv, dt = params.get_frames()
model = articulated.articulated_model(image.shape)

print image.shape

while(cv2.waitKey(10) != 'q'):

	# params.update()
	image, hsv, dt = params.get_frames()

	model.rz(0,1,0.02)
	model.rz(1,1,0.04)

	points = model.get_points()

	for i in range(len(points)):
		cv2.circle(image,(int(points[i][0]),int(points[i][1])),2,(0,0,255))

	points = model.get_points(all_points=True)

	for i in range(len(points)):
		cv2.circle(image,(int(points[i][0]),int(points[i][1])),1,(0,255,0))

	cv2.imshow("Neptune - Image", image)
	cv2.imshow("Neptune - HSV", hsv)
	cv2.imshow("Neptune - DT", dt)

image, hsv, dt = params.get_frames()

# points = model.get_points()

# for i in range(len(points)-1):
# 	if i != 4:
# 		cv2.line(image,(int(points[i][0]),int(points[i][1])),(int(points[i+1][0]),int(points[i+1][1])),(255,0,0))
# for i in range(len(points)):
# 	cv2.circle(image,(int(points[i][0]),int(points[i][1])),2,(0,0,255))

points = model.get_points(all_points=True)

for i in range(len(points)):
	print '-->', points[i]
	cv2.circle(image,(int(points[i][0]),int(points[i][1])),1,(0,255,0))

cv2.imshow("Neptune - HSV", hsv)
cv2.imshow("Neptune - DT", dt)
cv2.imshow("Neptune - Image", image)
cv2.waitKey(0)


























































