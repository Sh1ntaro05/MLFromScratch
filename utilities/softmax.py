import numpy as np

def softmax(a):
    if a.ndim == 1:
        c = np.max(a)
        exp_a = np.exp(a-c)
        return exp_a / np.sum(exp_a)
    c = np.max(a,axis=1,keepdims=True)
    exp_a = np.exp(a-c)
    return exp_a / np.sum(exp_a,axis=1,keepdims=True)