import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances

def eig_order(A, cluster_num):
    if cluster_num > A.shape[0]:
        cluster_num = A.shape[0]

    d, v = np.linalg.eigh(A)  #use eigh to avoid complex eigvectors
    idx = np.argsort(d) 
    idx1 = idx[:cluster_num]  
    eigval_cluster_num = d[idx1]  
    eigvec = v[:, idx1]  
    eigval_full_ordered = d[idx]  

    return eigvec, eigval_cluster_num, eigval_full_ordered

def projection_simplex(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    s = np.maximum(v - theta, 0)
    return s

def L2_distance_square(X, Y):
    # square euclidean distances
    return euclidean_distances(X, Y)**2


# X: Data matrix
# c: number of clusters
# k: number of neighbors
# islocal=1 means use neighbor to update matrix
# return y(cluster graph), S(similarity matrix)
def CAN(X, c, k, islocal=1):
    NITER = 30
    num = X.shape[0]

    distX = L2_distance_square(X, X)
    distX1 = np.sort(distX, axis=1)
    idx = np.argsort(distX, axis=1)

    # Initialize similarity matrix S 
    S = np.zeros((num, num))
    gamma = np.zeros(num)

    for i in range(num):
        di = distX1[i, 1:k+2] #k+1 columns nearest except itself
        gamma[i] = 0.5 * (k * di[-1] - np.sum(di[:-1])) #gammai (33)
        id = idx[i, 1:k+2]
        S[i, id] = (di[-1] - di) / (k * di[-1] - np.sum(di[:-1]) + np.finfo(float).eps) #kkt solving (3)

    r = np.mean(gamma)
    lambda_ = np.mean(gamma)

    # Initialize F
    S0 = (S + S.T) / 2
    D0 = np.diag(np.sum(S0, axis=1))
    L0 = D0 - S0

    F, _, evs = eig_order(L0, c)

    if np.sum(evs[:c+1]) < 1e-10:
        raise ValueError('More than %d connected components on original graph' % c)

    # Iterate
    for iter in range(NITER):
        distf = L2_distance_square(F, F)
        S = np.zeros((num, num))

        for i in range(num):
            if islocal == 1:
                idxa0 = idx[i, 1:k+1]
            else:
                idxa0 = np.arange(num)
            dfi = distf[i, idxa0]
            dxi = distX[i, idxa0]
            ad = -(dxi + lambda_ * dfi) / (2 * r) # (28)
            S[i, idxa0] = projection_simplex(ad,1) # (30)

        S = (S + S.T) / 2
        D = np.diag(np.sum(S, axis=1))
        L = D - S

        F_old = F
        F, _, ev = eig_order(L, c)
        evs = np.concatenate((evs, ev), axis=0)

        fn1 = np.sum(ev[:c])
        fn2 = np.sum(ev[:c+1])
        #update lamdba
        if fn1 > 1e-10:  #the connected components of S is smaller than c 
            lambda_ *= 2
        elif fn2 < 1e-10: #the connected components of S is larger than c 
            lambda_ /= 2
            F = F_old
        else:
            break

    # connect all the parts
    clusternum, y = connected_components(csgraph=csr_matrix(S), directed=False, return_labels=True)
    y = y + 1  

    if clusternum != c:
        print('Error with cluster number: %d' % c)

    return y, S
    