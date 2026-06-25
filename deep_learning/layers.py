import numpy as np
from utilities.cross_entropy_error import cross_entropy_error
from utilities.softmax import softmax 

class Layer:
    def __init__(self):
        pass 

    def forward(self, x, y):
        pass 

    def backward(self, dout):
        pass

    def get_params(self):
        pass

    def get_grads(self):
        pass


class MulLayer(Layer):
    def __init__(self):
        self.x = None 
        self.y = None 
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out 

    def backward(self, dout):
        dx = dout * self.x 
        dy = dout * self.y 
        return dx, dy 
    
    def get_params(self):
        return []

    def get_grads(self):
        return []

class AddLayer(Layer):
    def __init__(self):
        pass 

    def forward(self, x, y):
        out = x + y
        return out 

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1 
        return dx, dy 

    def get_params(self):
        return []
    
    def get_grads(self):
        return []
    
class ReLULayer(Layer):
    def __init__(self):
        self.mask = None 
    
    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0 
        return out 

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout 
        return dx 

    def get_params(self):
        return []
    
    def get_grads(self):
        return []

class SigmoidLayer(Layer): 
    def __init__(self):
        self.out = None 
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out 
        return out 

    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out 
        return dx 

    def get_params(self):
        return []
    
    def get_grads(self):
        return []

class AffineLayer(Layer):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None 
        self.dW = None 
        self.db = None 
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b 
        return out 

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

    def get_params(self):
        return [self.W, self.b]
    
    def get_grads(self):
        return [self.dW, self.db]

class SoftmaxWithLossLayer(Layer):
    def __init__(self):
        self.loss = None 
        self.y = None 
        self.t = None 
    
    def forward(self, x, t):
        self.t = t 
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss 
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx 
    
    def get_params(self):
        return []
    
    def get_grads(self):
        return []
        
    