import numpy as np 

def L_p_norm(x,y,p):
    abs_diff = np.abs(x-y)
    vec_sum = np.sum(np.power(abs_diff,p))
    return np.power(vec_sum,1/p)

def euclid_dist(x,y):
    return L_p_norm(x,y,2)

def manhattan_dist(x,y):
    return L_p_norm(x,y,1)

def chebichev_dist(x,y):
    abs_diff = np.abs(x-y)
    return np.max(abs_diff)