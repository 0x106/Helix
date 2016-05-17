import numpy as np
import cv2
import HSIC
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans

# A - affinity matrix
# C - clusters
def plot_points(A, C):
    # D = 1. - A
    # print D

    points = np.zeros((A.shape[0], 2))

    points[1,0] = A[0,1]

    a = A[0,1]
    b = A[0,2]
    c = A[1,2]

    print a, b, c

    points[2,0] = (a*a + b*b - c*c) / (2.*a)
    points[2,1] = np.sqrt(np.abs((a+b+c)*(a+b-c)*(b+c-a)*(c+a-b)) / (2.*a)) 

    print (a+b+c), (a+b-c), (b+c-a), (c+a-b)

    print points

    # plt.set_ylim(-20,100)
    # plt.set_xlim(-20,100)

    for i in range(A.shape[0]):
        plt.plot(points[i,:], 'o')
    plt.show()






def subsample_video(video, N, M):
    K = N/M#N - M + 1   # the number of output subsamples
    print N, M, K
    output = [np.zeros((video.shape[0], M)) for i in range(K)]

    for i in range(K):
        output[i] = video[:,i*M:(i+1)*M]    

    return output

def mocap_similarity():

    mocap_files = []

    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/16/16_'+ str(21)+ '.amc')   # 7, 4
    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/16/16_'+ str(22)+ '.amc')   # 7, 4
    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/16/16_'+ str(32)+ '.amc')   # 2, 7
    
    mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/01/01_0'+ str(2)+ '.amc')   # 0, 1
    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/01/01_0'+ str(3)+ '.amc')   # 0, 1
    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/01/01_0'+ str(4)+ '.amc')   # 0, 1

    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/09/09_0'+ str(2)+ '.amc')   # 1, 3
    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/09/09_0'+ str(3)+ '.amc')   # 1, 3
    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/09/09_0'+ str(4)+ '.amc')   # 1, 3

    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/08/08_0'+ str(2)+ '.amc')   # 4, 2
    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/08/08_0'+ str(3)+ '.amc')   # 6, 5
    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/08/08_0'+ str(4)+ '.amc')   # 3, 6

    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/05/05_0'+ str(2)+ '.amc')   # 0, 4
    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/05/05_0'+ str(3)+ '.amc')   # 0, 1
    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/05/05_0'+ str(4)+ '.amc')   # 5, 0

    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/26/26_0'+ str(1)+ '.amc')
    # mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/14/14_'+ str(20)+ '.amc')

    N, PCA_DIMS, M = 1000, 12, len(mocap_files)
    mocap_data = readData(mocap_files, N, _PCA=True)

    print mocap_data[0].shape

    video_subsamples = subsample_video(mocap_data[0], N, 50)

    M = len(video_subsamples)

    print len(video_subsamples)
    print video_subsamples[0].shape

    print '\n - \n - \n - \n - \n - \n - \n'

    print 'data dims:', video_subsamples[0].shape, mocap_data[0].shape

    # HS = HSIC.HilbertSchmidt(N)

    # covariance_data = np.zeros((PCA_DIMS, N*M))
    # for i in range(M):
    #     covariance_data[:,i*N:(i+1)*N] = mocap_data[i][:PCA_DIMS,:]

    HS = HSIC.HilbertSchmidt(100)
    covariance_data = np.zeros((PCA_DIMS, 100*M))
    for i in range(len(video_subsamples)):
        covariance_data[:,i*100:(i+1)*100] = video_subsamples[i][:PCA_DIMS,:]
    HS.set_covariance(covariance_data)

    print 'HSIC data dims:', video_subsamples[0][:PCA_DIMS, :].shape

    affinity = np.zeros((M,M))

    for i in range(M):
        for k in range(M):
            affinity[i,k] = HS.HSIC_AR(video_subsamples[i][:PCA_DIMS,:], video_subsamples[k][:PCA_DIMS,:])
            print affinity[i,k]
        print '--------------'

    plot2d(affinity)
    plt.show()


    clusters = sklearn.cluster.spectral_clustering(affinity, n_clusters=4)

    print clusters

    # if True:

    #     k = 3

    #     D = np.zeros((M,M))
    #     for i in range(M):
    #         D[i,i] = np.sum(affinity[i,:])
    #     D = np.linalg.inv(np.sqrt(D))

    #     L = np.dot(D, np.dot(affinity,D))


    #     w,v = np.linalg.eigh(L)
    #     idx = np.argsort(-w)
    #     v = v[:,idx]

    #     Y = (v[:,:k].T / (np.sum(v[:,:k],1))).T

    #     print Y

    #     estimator = KMeans(nclusters=2)


    #     e,EV = np.linalg.eigh(M)
    
    # tmp = np.dot(X.T,EV)
    # tmp = (tmp/np.sqrt(np.sum(tmp**2,0))).T
    
    # e[e<0] = 0
    # S = np.sqrt(e)
    
    # idx = np.argsort(-S)
    
    # S = S[idx]
    # V = tmp[idx,:]



def get_image_data(prefix, N, K):

    image = cv2.imread(prefix + str(1)+'.png')
    h,w = image.shape[:2]

    data__ = np.zeros((h*w*3, K))

    for i in range(K):
        image = cv2.imread(prefix + str(i+1)+'.png', cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2HSV)
        # data__[:,i] = np.asarray(image)[:,:,:].flatten()
        data__[:,i] = np.asarray(hsv)[:,:,:].flatten()

    index = np.linspace(0, K-1, num=N)

    # data = np.zeros((h*w*3, N))

    data = PCA(data__[:,index.astype(int)].T)

    return data

def readData(files, N, _PCA=False):
    output = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/data/temp.amc'
    data = []
    DOF = 0
    for fidx, f in enumerate(files):
        file = open(f)
        out = open(output, 'w')
        lines = [line.split() for line in file]
        print len(lines)
        # N = len(lines)
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

        data[fidx] = data[fidx].T

        if _PCA:
            data[fidx] = PCA(data[fidx].T)
    return data

def run():

    mocap_files = []
    image_files = []
    mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/01/01_0'+ str(1)+ '.amc')
    mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/01/01_0'+ str(2)+ '.amc')
    mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/26/26_0'+ str(1)+ '.amc')
    mocap_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/all_asfamc/subjects/14/14_'+ str(20)+ '.amc')

    image_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/01/01/')
    image_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/01/02/')
    image_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/26/01/')
    image_files.append('/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CMU/14/20/')

    # video_subsamples = subsample_video(image_files[0], N=1, K=750, M=250)                
    
    image_data = get_image_data(image_files[0], N, 750)

    M = 250

    

    N = len(video_subsamples)

    results = np.zeros((N,N))

    image = cv2.imread(image_files[0] + str(1)+'.png')
    h,w = image.shape[:2]

    HS = HSIC.HilbertSchmidt(250,h*w*3, 20.)

    print 'N -->', N

    for i in range(N):
        for k in range(N):
            a,b,c,d = HS.HSIC(video_subsamples[i], video_subsamples[k], h*w*3, 20., 20.)
            results[i,k] = d


    plot2d(results)
    plt.show()
















    N = 250

    image_data = []

    mocap_data = readData(mocap_files, N)
    
    image_data.append(get_image_data(image_files[0], N, 750)   / 1e3)
    image_data.append(get_image_data(image_files[1], N, 1143)  / 1e3)
    image_data.append(get_image_data(image_files[2], N, 251)   / 1e3)
    image_data.append(get_image_data(image_files[3], N, 1210)   / 1e3)

    PCA = 20
    HS = HSIC.HilbertSchmidt(N, _K=PCA, _Q=400.)

    print 'mocap | image'
    a, b, c, d = HS.HSIC(mocap_data[0][:PCA,:] / 10, image_data[0][:PCA,:], PCA, 400., 160.)
    print '0 - 0', a,b,c,d
    a, b, c, d = HS.HSIC(mocap_data[0][:PCA,:] / 10, image_data[1][:PCA,:], PCA, 400., 160.)
    print '0 - 0', a,b,c,d
    a, b, c, d = HS.HSIC(mocap_data[0][:PCA,:] / 10, image_data[2][:PCA,:], PCA, 400., 160.)
    print '0 - 0', a,b,c,d
    # a, b, c, d = HS.HSIC(mocap_data[0][:PCA,:] / 10, image_data[3][:PCA,:], PCA, 400., 160.)
    # print '0 - 0', a,b,c,d

    a, b, c, d = HS.HSIC(mocap_data[1][:PCA,:] / 10, image_data[0][:PCA,:], PCA, 1400., 160.)
    print '0 - 0', a,b,c,d
    a, b, c, d = HS.HSIC(mocap_data[1][:PCA,:] / 10, image_data[1][:PCA,:], PCA, 1400., 160.)
    print '0 - 0', a,b,c,d
    a, b, c, d = HS.HSIC(mocap_data[1][:PCA,:] / 10, image_data[2][:PCA,:], PCA, 1400., 160.)
    print '0 - 0', a,b,c,d
    # a, b, c, d = HS.HSIC(mocap_data[1][:PCA,:] / 10, image_data[3][:PCA,:], PCA, 1400., 160.)
    # print '0 - 0', a,b,c,d

    a, b, c, d = HS.HSIC(mocap_data[2][:PCA,:] / 10, image_data[0][:PCA,:], PCA, 400., 160.)
    print '0 - 0', a,b,c,d
    a, b, c, d = HS.HSIC(mocap_data[2][:PCA,:] / 10, image_data[1][:PCA,:], PCA, 400., 160.)
    print '0 - 0', a,b,c,d
    a, b, c, d = HS.HSIC(mocap_data[2][:PCA,:] / 10, image_data[2][:PCA,:], PCA, 400., 160.)
    print '0 - 0', a,b,c,d
    # a, b, c, d = HS.HSIC(mocap_data[2][:PCA,:] / 10, image_data[3][:PCA,:], PCA, 400., 160.)
    # print '0 - 0', a,b,c,d

    a, b, c, d = HS.HSIC(mocap_data[3][:PCA,:] / 10, image_data[0][:PCA,:], PCA, 400., 160.)
    print '0 - 0', a,b,c,d
    a, b, c, d = HS.HSIC(mocap_data[3][:PCA,:] / 10, image_data[1][:PCA,:], PCA, 400., 160.)
    print '0 - 0', a,b,c,d
    a, b, c, d = HS.HSIC(mocap_data[3][:PCA,:] / 10, image_data[2][:PCA,:], PCA, 400., 160.)
    print '0 - 0', a,b,c,d
    # a, b, c, d = HS.HSIC(mocap_data[3][:PCA,:] / 10, image_data[3][:PCA,:], PCA, 400., 160.)
    # print '0 - 0', a,b,c,d
    # print '0 - 1', HS.HSIC(mocap_data[0][:PCA,:] / 10, image_data[1][:PCA,:], PCA, 200., 200.)
    # print '0 - 2', HS.HSIC(mocap_data[0][:PCA,:] / 10, image_data[2][:PCA,:], PCA, 200., 200.)

    # print '1 - 0', HS.HSIC(mocap_data[1][:PCA,:] / 10, image_data[0][:PCA,:], PCA, 200., 200.)
    # print '1 - 1', HS.HSIC(mocap_data[1][:PCA,:] / 10, image_data[1][:PCA,:], PCA, 200., 200.)
    # print '1 - 2', HS.HSIC(mocap_data[1][:PCA,:] / 10, image_data[2][:PCA,:], PCA, 200., 200.)

    # print '2 - 0', HS.HSIC(mocap_data[2][:PCA,:] / 10, image_data[0][:PCA,:], PCA, 200., 200.)
    # print '2 - 1', HS.HSIC(mocap_data[2][:PCA,:] / 10, image_data[1][:PCA,:], PCA, 200., 200.)
    # print '2 - 2', HS.HSIC(mocap_data[2][:PCA,:] / 10, image_data[2][:PCA,:], PCA, 200., 200.)

    plt.subplot(241)
    plt.plot(mocap_data[0][0,:])
    plt.plot(mocap_data[0][1,:])
    plt.plot(mocap_data[0][2,:])

    plt.subplot(242)
    plt.plot(mocap_data[1][0,:])
    plt.plot(mocap_data[1][1,:])
    plt.plot(mocap_data[1][2,:])

    plt.subplot(243)
    plt.plot(mocap_data[2][0,:])
    plt.plot(mocap_data[2][1,:])
    plt.plot(mocap_data[2][2,:])

    plt.subplot(244)
    plt.plot(mocap_data[3][0,:])
    plt.plot(mocap_data[3][1,:])
    plt.plot(mocap_data[3][2,:])

    plt.subplot(245)
    plt.plot(image_data[0][0,:])
    plt.plot(image_data[0][1,:])
    plt.plot(image_data[0][2,:])

    plt.subplot(246)
    plt.plot(image_data[1][0,:])
    plt.plot(image_data[1][1,:])
    plt.plot(image_data[1][2,:])

    plt.subplot(247)
    plt.plot(image_data[2][0,:])
    plt.plot(image_data[2][1,:])
    plt.plot(image_data[2][2,:])

    plt.subplot(248)
    plt.plot(image_data[3][0,:])
    plt.plot(image_data[3][1,:])
    plt.plot(image_data[3][2,:])

    plt.show()


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
    
    X += mu_X

    return Y


def CHG_match():

    prefix = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CHG/set1/'

    data = get_opt_data(prefix, IMAGE=True)

    print data[0].shape

    N, M, PCA_DIMS = 59, len(data), 20

    HS = HSIC.HilbertSchmidt(N)

    covariance_data = np.zeros((PCA_DIMS, N*M))
    for i in range(M):
        covariance_data[:,i*N:(i+1)*N] = data[i][:PCA_DIMS,:]

    HS.set_covariance(covariance_data)

    affinity = np.zeros((M,M))

    for i in range(M):
        for k in range(M):
            affinity[i,k] = HS.HSIC_AR(data[i][:PCA_DIMS,:], data[k][:PCA_DIMS,:])
            print affinity[i,k]
        print '--------------'

    plot2d(affinity)
    plt.show()


def get_opt_data(prefix, IMAGE=False):

    suffix_1 = '/frame-'
    suffix_2 = '.jpg'

    L1,L2,count,mfc = 9,3,0,60
    N = mfc-1

    data = []
    image_data = []

    for i in range(L1):
        for j in range(L2):
            count_file = open(prefix + str(i).zfill(4) + '/'+str(j).zfill(4)+'/count.txt')
            for line in count_file:
                count = line.split()[0]

            index = np.linspace(0,int(count)-1, num=mfc)

            tensor_flow = []

            if IMAGE:
                prev = cv2.imread(prefix+str(0).zfill(4)+'/' +str(0).zfill(4)+suffix_1+str(0).zfill(4)+'.jpg', 0)
                data.append(np.zeros((prev.shape[0]*prev.shape[1]*4, N)))
    
            print prefix + str(i).zfill(4)+'/'+str(j).zfill(4)

            for k in range(N):
             
                prev_file = prefix + str(i).zfill(4)+'/'+str(j).zfill(4)+ suffix_1+str(int(index[k])).zfill(4)+'.jpg'
                next_file = prefix + str(i).zfill(4)+'/'+str(j).zfill(4)+ suffix_1+str(int(index[k+1])).zfill(4)+'.jpg'
                
                prev = cv2.imread(prev_file, 0)
                next = cv2.imread(next_file, 0)
                frame = cv2.imread(prev_file, cv2.IMREAD_COLOR)

                if IMAGE:
                    edges = cv2.Canny(np.copy(prev), 50,150)
                    dt = cv2.distanceTransform(np.copy(edges), cv2.DIST_L2, maskSize=5)
                    cv2.normalize(np.copy(dt), dt, alpha=0., beta=1.0, norm_type=cv2.NORM_MINMAX)
                    hsv = cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2HSV)

                    data[-1][:prev.shape[0]*prev.shape[1],k] = np.asarray(dt)[:,:].flatten()
                    # data[-1][prev.shape[0]*prev.shape[1]:prev.shape[0]*prev.shape[1]*4,k] = np.asarray(frame)[:,:,:].flatten()
                    data[-1][prev.shape[0]*prev.shape[1]:,k] = np.asarray(hsv)[:,:,:].flatten()

                else:
                    flow = cv2.calcOpticalFlowFarneback(prev,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)                
                    tensor_flow.append(flow)

            if not IMAGE:
                flow_img, tracks, data__ = show_tracks(tensor_flow, frame, N)
                data.append(data__)

            print data[-1].shape

            data[-1] = PCA(data[-1].T)

    return data



def optical_flow_tracking():

    dir_prefix = '/Users/jordancampbell/Desktop/Helix/code/pyNeptune/data/CHG/set1/'
    count = 0

    # this allows all the flow tensors to be built from the same number of frames
    L1, L2, mfc = 9, 5, 40
    N = mfc-1
    M = 2400
    F = 20

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

            prev_file = dir_prefix +str(level_one).zfill(4)+'/'+str(level_two).zfill(4)+'/'+'frame-'+str(0).zfill(4)+'.jpg'
            prev = cv2.imread(prev_file, 0)
            IMAGE_data = np.zeros((prev.shape[0]*prev.shape[1]*3, N))

            for k in range(N):
             
                prev_file = dir_prefix +str(level_one).zfill(4)+'/'+str(level_two).zfill(4)+'/'+'frame-'+str(int(index[k])).zfill(4)+'.jpg'
                next_file = dir_prefix +str(level_one).zfill(4)+'/'+str(level_two).zfill(4)+'/'+'frame-'+str(int(index[k+1])).zfill(4)+'.jpg'
                
                prev = cv2.imread(prev_file, 0)
                next = cv2.imread(next_file, 0)
                frame = cv2.imread(prev_file, cv2.IMREAD_COLOR)

                flow = cv2.calcOpticalFlowFarneback(prev,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # flow_img = draw_flow(frame, flow)

                tensor_flow[-1].append(flow)

                # IMAGE_data[:prev.shape[0]*prev.shape[1]*3,k] = np.asarray(frame)[:,:,:].flatten()
                hsv = cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2HSV)
                IMAGE_data[:,k] = np.asarray(hsv)[:,:,:].flatten()
                # cv2.imshow("CHG Dataset - Flow", flow_img)
                # cv2.waitKey(1)

            flow_img, tracks, data = show_tracks(tensor_flow[-1], frame, N)

            data = PCA(data.T) / 1.e2
            IMAGE_data = PCA(IMAGE_data.T) / 1.e3

            # plt.subplot(121)
            # plt.plot(data[0,:])
            # plt.plot(data[1,:])
            # plt.plot(data[2,:])
            # plt.plot(data[3,:])

            # plt.subplot(122)
            # plt.plot(IMAGE_data[0,:])
            # plt.plot(IMAGE_data[1,:])
            # plt.plot(IMAGE_data[2,:])
            # plt.plot(IMAGE_data[3,:])
            # plt.plot(IMAGE_data[0,:],'+')
            # plt.plot(IMAGE_data[1,:],'+')
            # plt.plot(IMAGE_data[2,:],'+')
            # plt.plot(IMAGE_data[3,:],'+')
            # plt.show()

            for i in range(N):
                for k in range(N):
                    # KH[file_index][i,k] = HS.rbf(data[:,i], data[:,k])
                    KH[file_index][i,k] = HS.rbf(IMAGE_data[:F,i], IMAGE_data[:F,k])

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



