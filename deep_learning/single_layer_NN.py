import numpy as np
import pandas as pd

class Node:
    def __init__(self,bias:float=None,weights:np.ndarray=None,value=None,num_weights=None):
        self.bias = None
        self.weights = None
        self.value = None 
        self.num_weights = None
        pass 

class Layer:
    def __init__(self,bias_vec:np.ndarray=None,weights_matrix:np.ndarray=None,values:np.ndarray=None,num_nodes:np.ndarray=None,num_inputs:np.ndarray=None):
        self.bias_vec = None 
        self.weights_matrix = None 
        self.values = None 
        self.num_nodes = None 
        self.num_inputs = None 
        self.nodes = None 
        pass 

class SingleLayerNN:
    def __init__(self):
        self.n = None
        self.p = None
        self.output_node = None 
        self.hidden_layer = None 
        self.output_layer = None
        self.num_hidden_nodes = None
        self.batch_size = None
        self.epochs = None
        pass

    def fit(self,X:np.ndarray,y:np.ndarray,num_hidden_nodes:int,batch_size:int,epochs:int) -> None:  
        self.n = X.shape[0]
        self.p = X.shape[1]
        #whole_data = np.hstack((X,y))
        #X_design = np.empty((self.data_n,X.shape[1]+1))
        #X_design[:,0] = 1.0
        #X_design[:,1:] = X
        self.num_hidden_nodes = num_hidden_nodes
        #self.output_node = Node(bias=0.0,weights = np.zeros((self.num_hidden_nodes,1)))
        self.hidden_layer = Layer(
            bias_vec=np.zeros((self.num_hidden_nodes,1)),
            weights_matrix=np.zeros((self.num_hidden_nodes,self.p)),
            values=np.zeros((self.num_hidden_nodes,1)),
            num_nodes=self.num_hidden_nodes,
            num_inputs=self.p
            )
        """ self.output_layer = Layer(
            bias_vec = np.array([0.0]),
            weights_matrix=np.zeros((1,self.num_hidden_nodes)),
            num_nodes=1,
            num_inputs=self.num_hidden_nodes
        ) """
        self.output_node= Node(
            bias = 0.0,
            weights=np.zeros((self.num_hidden_nodes,1))
        )
        self.batch_size = batch_size
        self.epochs = epochs

        for epoch in self.epochs:
            shuffled_indices = np.random.permutation(self.n)
            for start_idx in range(0,self.n,self.batch_size):
                loss_sum = 0.0
                end_idx = start_idx + self.batch_size
                batch_indices = shuffled_indices[start_idx:end_idx]
                batch = X[batch_indices]
                y_vec = y[batch_indices]
                y_hat_vec = self.forward_pass(batch)
                loss_sum += self.loss_function(y_hat_vec,y_vec)
             

        pass

    def predict(self,X:np.ndarray):




        pass 


    #Recieves: A vector
    #Does: Calculates the RELU value of the whole vector
    #Returns: RELU(vec)
    def RELU(self, vec:np.ndarray) -> np.ndarray:
        mask = vec > 0
        return np.where(mask == 0, 0.0, vec)
    
    def D_RELU(self,vec:np.ndarray) -> np.ndarray:
        return np.where(vec < 0, 0, 1.0)
    
    #Recieves: A batch, which is a submatrix of the entire data matrix, with each row being one data point
    #Does: For the entire batch, gives a prediction for each data point using the current values of weights and biases
    #Returns: A vector, which is the predicted values for each data point
    def forward_pass(self,batch:np.ndarray) -> np.ndarray:
        Z = self.hidden_layer.weights_matrix @ batch.T + self.hidden_layer.reshape((len(self.hidden_layer.weights_matrix),1))
        pred_y = self.output_node.bias + self.output_node.weights @ self.RELU(Z).T 
        return pred_y

    def loss_function(self,y_hat_vec,y_vec):
        return 0.5 * (y_vec - y_hat_vec).T @ (y_vec - y_hat_vec) 
    
    def D_loss_function(self,y_hat_vec,y_vec):
        return y_hat_vec - y_vec
        
def main():
    
    pass 

if __name__ == "__main__":
    main()
