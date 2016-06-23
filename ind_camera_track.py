import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_flow(im, flow, step=4):

    h,w = im.shape[:2]

    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)

    seeds = [int(np.random.rand() * w), int(np.random.rand() * h)]

    temp_flow = flow - flow[seeds[1], seeds[0]]

    fx,fy = temp_flow[y,x].T

    lines = np.vstack([x,y,x+fx, y+fy]).T.reshape(-1,2,2)
    lines = lines.astype(int)

    for (x1,y1), (x2,y2) in lines:
        if (np.sqrt(((x2-x1)*(x2-x1)) + ((y2-y1)*(y2-y1)))) < 0.5:
            # cv2.line(im, (x1,y1),(x2,y2),(0,255,0),1)
            cv2.circle(im,(x1,y1),1,(0,255,0), -1)
        else:
            # cv2.line(im, (x1,y1),(x2,y2),(0,0,255),1)
            cv2.circle(im,(x1,y1),1,(0,0,255), -1)

    return im

dir = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/independent_moving_cameras/'
start_frame = 1

for idx in range(start_frame, start_frame + 500):

	prevImg = cv2.imread(dir + str(idx) + '.png', 0)
	nextImg = cv2.imread(dir + str(idx+1) + '.png', 0)

	frame = cv2.imread(dir + str(idx) + '.png', cv2.IMREAD_COLOR)

	flow = cv2.calcOpticalFlowFarneback(prevImg,nextImg, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	flow_img = draw_flow(frame, flow)

	# cv2.imshow('Previous', prevImg)
	# cv2.imshow('Next', nextImg)
	cv2.imshow('Flow', flow_img)

	cv2.waitKey(1)
