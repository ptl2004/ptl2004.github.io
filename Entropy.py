import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage

def Entropy(X):
    """Calculate entropy from positions"""
    Y = pdist(X)
    R = squareform(Y)
    Z = linkage(X, method='single')
    
    h = Z[:, 2].T - 0.0001
    Entro = 0.0
    n_points = R.shape[0]
    Robots = list(range(n_points))
    
    for d in range(len(h)):
        c = {}
        
        for i in Robots:
            c[i] = [i]
            Excepti = [r for r in Robots if r != i]
            
            for j in Excepti:
                count = 0
                for k in c[i]:
                    if R[j, k] <= h[d]:
                        count += 1
                
                if count == len(c[i]):
                    c[i].append(j)
        
        # Discard redundant clusters
        sumci = len(c[0])
        cip = c[0].copy()
        
        for i in range(1, n_points):
            if i not in cip:
                sumci += len(c[i])
                cip.extend(c[i])
            else:
                c[i] = []
        
        Hd = 0.0
        for i in Robots:
            if c[i]:  # if not empty
                Pj = len(c[i]) / sumci
                Hd = Hd - Pj * np.log2(Pj)
        
        if d == 0:
            Entro = Entro + Hd * h[d]
        else:
            Entro = Entro + Hd * (h[d] - h[d-1])
    
    return Entro