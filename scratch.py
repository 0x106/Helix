import numpy as np
import time

def test():

    N = 200

    A = np.random.rand(N,N)
    B = np.random.rand(N,N)
    C = np.random.rand(N,N)

    X = np.zeros((N,N))
    Y = np.zeros((N,N))
    Z = np.zeros((N,N))

    x_start = time.clock()
    for i in range(N):
        for k in range(N):
            X[i,k] = np.exp(-(np.dot((A[:,i]-B[:,k]).T, np.dot(C, A[:,i]-B[:,k]))))
    x_stop = time.clock()

    y_start = time.clock()
    for i in range(N):
        Y[i,:] = np.diag(np.exp(-(np.dot(  (A[:,i] - B.T)  ,  (np.dot(C, (A[:,i] - B.T).T).T).T  ) )))
    y_stop = time.clock()

    z_start = time.clock()
    for i in range(N):
        Z[i,:] = np.exp(-np.sum((A[:,i] - B.T)*((np.dot(C, (A[:,i] - B.T).T).T).T).T, axis=1))
    z_stop = time.clock()

    print 'Time:\t', x_stop - x_start, '\t',y_stop - y_start, '\t',z_stop - z_start
    print 'Error:\t', np.abs(np.round(np.sum(X - Y))), '\t','\t',np.abs(np.round(np.sum(X - Z))), '\t','\t',np.abs(np.round(np.sum(Y - Z)))
    print 'Error:\t', np.abs(np.sum(X - Y)), '\t','\t',np.abs(np.sum(X - Z)), '\t','\t',np.abs(np.sum(Y - Z))

test()

if False:

    #A = np.random.rand(3,3)
    #B = np.random.rand(3,3)

    A = np.zeros((3,3))
    B = np.zeros((3,3))
    C = np.random.rand(3,3)

    A[0,0] = 1
    A[0,1] = 1
    A[0,2] = 1

    A[1,0] = 1
    A[1,1] = 1
    A[1,2] = 1

    A[2,0] = 1
    A[2,1] = 1
    A[2,2] = 1

    B[0,0] = 10
    B[0,1] = 11
    B[0,2] = 12

    B[1,0] = 13
    B[1,1] = 14
    B[1,2] = 15

    B[2,0] = 16
    B[2,1] = 17
    B[2,2] = 18

    #print A
    #print B
    #print C

    #print '---------------------'
    #print A[:,0] - B

    #print '---------------------'


    #print A[:,0] - B[:,0]
    #print A[:,0] - B[:,1]
    #print A[:,0] - B[:,2]

    #print '---------------------'

    #print np.dot(C, A[:,0] - B)#
    #print '---------------------'

    #print np.dot(C, A[:,0] - B[:,0])
    #print np.dot(C, A[:,0] - B[:,1])
    #print np.dot(C, A[:,0] - B[:,2])
