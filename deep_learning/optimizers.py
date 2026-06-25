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
        pass


class AdaGrad(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr 
        self.h = None 
    
    def update(self, params, grads):
        pass 


class Adam(Optimizer):
    pass