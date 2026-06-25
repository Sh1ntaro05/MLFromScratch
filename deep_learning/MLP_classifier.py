import numpy as np
from typing import List

import deep_learning.layers as layers
import deep_learning.optimizers as optimizers

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class MLPClassifier:
    def __init__(self,layers_list:List[layers.Layer],random_state=42):
        self.layers_list = layers_list[:-1]
        self.lastlayer = layers_list[-1]
        self.random_state = random_state

    def predict(self, x):
        for layer in self.layers_list:
            x = layer.forward(x)
        return x
    
    def get_loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y,t)
        
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy 
    
    def backward_pass(self, x, t):
        self.get_loss(x, t)
        dout = 1.0
        dout = self.lastlayer.backward(dout)

        for layer in reversed(self.layers_list):
            dout = layer.backward(dout)
        
        pass 

    def get_params(self):
        all_params = []
        for layer in self.layers_list:
            all_params.extend(layer.get_params())
        return all_params
    
    def get_grads(self):
        all_grads = []
        for layer in self.layers_list:
            all_grads.extend(layer.get_grads())
        return all_grads

    def fit(self, optimizer:optimizers.Optimizer, epochs, batch_size, x, t):
        data_size = x.shape[0]
        iters = int(data_size / batch_size)
        iter_per_epoch = max(data_size / batch_size, 1)
        
        for epoch in range(epochs):
            index = np.random.permutation(data_size)
            for i in range(iters):
                batch_index = index[i * batch_size : (i+1) * batch_size]
                x_batch = x[batch_index]
                t_batch = t[batch_index]

                #loss = self.get_loss(x_batch,t_batch)
                self.backward_pass(x_batch, t_batch)

                grads = self.get_grads()
                params = self.get_params()

                optimizer.update(params,grads)

                #sum_loss += loss

            if epoch % iter_per_epoch == 0:
                loss = self.get_loss(x_batch, t_batch)
                acc = self.accuracy(x_batch,t_batch)
                print(f"Loss: {loss} | Accuracy: {acc}")

def get_mnist_data():
    print("Downloading MNIST... (this might take a minute)")
    # fetch_openml downloads the dataset and caches it locally
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # 1. Normalization
    # Images are flattened arrays of 784 pixels (28x28). Raw values are 0 to 255.
    X = X / 255.0
    
    # 2. One-Hot Encoding the Labels
    # 'y' comes down as an array of strings ['5', '0', '4', ...]. 
    y = y.astype(int)
    data_size = y.size
    t = np.zeros((data_size, 10))
    t[np.arange(data_size), y] = 1 # Advanced NumPy indexing to set the correct class to 1
    
    # 3. Train/Test Split
    # We reserve 10,000 images for testing our accuracy after training
    x_train, x_test, t_train, t_test = train_test_split(X, t, test_size=10000, random_state=42)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {t_train.shape}")
    
    return x_train, x_test, t_train, t_test


def main():
    x_train, x_test, t_train, t_test = get_mnist_data()

    initial_std = 0.01
    input_size = 784
    hidden_size = 50
    output_size = 10

    W1 = initial_std * np.random.randn(input_size, hidden_size)
    b1 = np.zeros(hidden_size)
    affine1 = layers.AffineLayer(W1,b1)
    
    W2 = initial_std * np.random.randn(hidden_size, output_size)
    b2 = np.zeros(output_size)
    affine2 = layers.AffineLayer(W2,b2)

    relu1 = layers.ReLULayer()
    
    softmax = layers.SoftmaxWithLossLayer()
    
    layers_list = [affine1,relu1,affine2,softmax]

    model = MLPClassifier(layers_list)
    optimizer = optimizers.Momentum()
    model.fit(optimizer,100,1000,x_train,t_train)

    print(f"Accuracy when tested on test data: {model.accuracy(x_test, t_test)}")
    
    pass

if __name__ == "__main__":
    main()



        



