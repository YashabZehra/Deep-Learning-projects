import numpy as np

def cross_entropy(Y,P):
    y=np.float64(Y)
    p=np.float64(P)

    return -np.sum(y*np.log(p)+(1-y)*np.log(1-p))  #sum(ylogp+(1-y)log(1-p))  y=actual output, p=predicted output



print(cross_entropy([1,0],[0.9,0.6]))