import numpy as np
import cv2
import HSIC
import matplotlib.pyplot as plt

def readData(files, N):
    output = '/Users/jordancampbell/Desktop/all_asfamc/data/temp.amc'
    data = []
    DOF = 0
    for fidx, f in enumerate(files):
        file = open(f)
        out = open(output, 'w')
        lines = [line.split() for line in file]
        for i in range(3, len(lines)):
            if len(lines[i]) == 1:
                for k in range(1,28):
                    for j in range(1, len(lines[i+k])):
                        out.write(lines[i+k][j])
                        out.write(' ')
                out.write('\n')
        out.close()
        file = open(output)
        lines = [[float(x) for x in line.split()] for line in file]
        data.append(np.zeros((N, len(lines[0]))))
        index = np.linspace(0, len(lines)-1, num=N).astype(int)
        DOF = len(lines[0])
        for i in range(N):
            for k in range(len(lines[0])):
                data[fidx][i,k] = lines[index[i]][k]
    return data

def run():

    files = []
    files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/08/08_0'+ str(1)+ '.amc')
    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/08/08_0'+ str(2)+ '.amc')
    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/08/08_0'+ str(3)+ '.amc')
    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/08/08_0'+ str(4)+ '.amc')
    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/08/08_0'+ str(5)+ '.amc')

    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/107/08_0'+ str(1)+ '.amc')
    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/107/08_0'+ str(2)+ '.amc')
    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/107/08_0'+ str(3)+ '.amc')
    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/08/08_0'+ str(4)+ '.amc')
    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/08/08_0'+ str(5)+ '.amc')

    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/08/08_0'+ str(1)+ '.amc')
    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/08/08_0'+ str(2)+ '.amc')
    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/08/08_0'+ str(3)+ '.amc')
    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/08/08_0'+ str(4)+ '.amc')
    # files.append('/Users/jordancampbell/Desktop/all_asfamc/subjects/08/08_0'+ str(5)+ '.amc')

    N = 100

    data = readData(files, N)

    print len(data)

def draw_flow(im, flow, step=8):

    h,w = im.shape[:2]

    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T

    lines = np.vstack([x,y,x+fx, y+fy]).T.reshape(-1,2,2)
    lines = lines.astype(int)

    for (x1,y1), (x2,y2) in lines:
        cv2.line(im, (x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(im,(x1,y1),1,(0,255,0), -1)
    return im

def show_tracks(flow_tensor, image, frame, save=False):

    step = 8
    h,w = image.shape[:2]

    pt = np.zeros(2)
    pt2 = np.zeros(2)

    x,y = np.linspace(0+step/2.,w-step/2.,w/step).astype(int), np.linspace(0+step/2.,h-step/2.,h/step).astype(int)

    tracks = np.zeros((x.shape[0]*y.shape[0],frame, 2))
    data = np.zeros((x.shape[0]*y.shape[0]*2., frame))

    index = 0
    idx = 0
    for i in range(x.shape[0]):
        for k in range(y.shape[0]):

            pt[:] = x[i],y[k]

            for j in range(frame):

                if pt[0] < 0:
                     pt[0] = 0
                if pt[1] < 0:
                     pt[1] = 0
                if pt[0] > w:
                     pt[0] = w-1
                if pt[1] > h:
                     pt[1] = h-1

                pt2 = pt + flow_tensor[j][pt[1].astype(int), pt[0].astype(int)]

                cv2.line(image, (pt[0].astype(int), pt[1].astype(int)), 
                    (pt2[0].astype(int), pt2[1].astype(int)) ,(0,0,255))

                tracks[index,j,:] = pt2[0].astype(int), pt2[1].astype(int)

                pt = np.copy(pt2)

            index += 1

    for j in range(frame):

        index = 0

        for k in range(tracks.shape[0]):

            data[index + 0, j] = tracks[k,j,0]
            data[index + 1, j] = tracks[k,j,1]

            index += 2

    return image, tracks, data

def plot2d(input):
    im = plt.imshow(input, interpolation='nearest', cmap='spring')
    plt.gca().invert_yaxis()
    plt.colorbar()

def PCA(X):
    mu_X = np.mean(X,0)
    X -= mu_X
    M = np.dot(X,X.T)
    e,EV = np.linalg.eigh(M)
    
    tmp = np.dot(X.T,EV)
    tmp = (tmp/np.sqrt(np.sum(tmp**2,0))).T
    
    e[e<0] = 0
    S = np.sqrt(e)
    
    idx = np.argsort(-S)
    
    S = S[idx]
    V = tmp[idx,:]
    
    Y = np.dot(V,X.T)
    
    return Y

def optical_flow_tracking():

    dir_prefix = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CHG/'
    count = 0

    # this allows all the flow tensors to be built from the same number of frames
    L1, L2, mfc = 9, 5, 40
    N = mfc-1
    M = 2400
    F = 10

    # from each video we extract a list of flow matrices
    # therefore our flow tensor is a list of list of matrices
    tensor_flow = [[]]

    KH = [np.zeros((mfc-1, mfc-1)) for i in range(L1*L2)]

    HS = HSIC.HilbertSchmidt(N,F)#M)

    file_index = 0
    for level_one in range(L1):
        for level_two in range(L2):

            print level_one, level_two, L1, L2

            cf = open(dir_prefix +str(level_one).zfill(4)+'/'+str(level_two).zfill(4)+'/'+'count.txt')
            for line in cf:
                count = line.split()[0]

            index = np.linspace(0, int(count)-1, num=mfc)

            tensor_flow.append([])

            for k in range(N):
             
                prev_file = dir_prefix +str(level_one).zfill(4)+'/'+str(level_two).zfill(4)+'/'+'frame-'+str(int(index[k])).zfill(4)+'.jpg'
                next_file = dir_prefix +str(level_one).zfill(4)+'/'+str(level_two).zfill(4)+'/'+'frame-'+str(int(index[k+1])).zfill(4)+'.jpg'
                
                prev = cv2.imread(prev_file, 0)
                next = cv2.imread(next_file, 0)
                frame = cv2.imread(prev_file, cv2.IMREAD_COLOR)

                flow = cv2.calcOpticalFlowFarneback(prev,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # flow_img = draw_flow(frame, flow)

                tensor_flow[-1].append(flow)

                # cv2.imshow("CHG Dataset - Flow", flow_img)
                # cv2.waitKey(1)

            flow_img, tracks, data = show_tracks(tensor_flow[-1], frame, N)

            data = PCA(data.T)

            # print data.shape

            # plt.plot(data[0,:])
            # plt.plot(data[1,:])
            # plt.plot(data[2,:])
            # plt.plot(data[3,:])
            # plt.show()

            for i in range(N):
                for k in range(N):
                    # KH[file_index][i,k] = HS.rbf(data[:,i], data[:,k])
                    KH[file_index][i,k] = HS.rbf(data[:F,i], data[:F,k])

            KH[file_index] = np.dot(KH[file_index], HS.get_H())

            # frame = cv2.imread(prev_file)

            # for i in range(tracks.shape[0]):
                # for k in range(tracks.shape[1]-1):
                #     cv2.line(frame, (tracks[i,k,0].astype(int), tracks[i,k,1].astype(int)), 
                #         (tracks[i,k+1,0].astype(int), tracks[i,k+1,1].astype(int)) ,(255,0,0))

            # cv2.imshow("Flow", frame)
            # cv2.waitKey(1)

            file_index += 1

    print len(KH)

    Q = len(KH)

    results = np.zeros((Q,Q))

    for i in range(Q):
        for k in range(Q):
            results[i,k] = (1./(N*N)) * np.trace(np.dot(KH[i], KH[k])) / np.sqrt(((1./(N*N)) * np.trace(np.dot(KH[k], KH[k]))) * ((1./(N*N)) * np.trace(np.dot(KH[i], KH[i]))))


    plot2d(results)


    # for i in range(Q):
        # plt.plot(results[i,:])

    plt.show()



