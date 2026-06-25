import numpy as np

class Optimizer:
    def __init__(self):
        pass 

    def update(self, params, grads):
        pass 


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = 0.01

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
        pass


class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr 
        self.momentum = momentum 
        self.v = None 

    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for i in range(len(params)):
                self.v.append(np.zeros_like(params[i]))
        
        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]
    
        pass


class AdaGrad(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr 
        self.h = None 
    
    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for i in range(len(params)):
                self.h.append(np.zeros_like(params[i]))
        
        for i in range(len(params)):
            self.h[i] += grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)

        pass 


class Adam(Optimizer):
    pass