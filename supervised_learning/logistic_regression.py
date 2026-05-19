import numpy as np

class LogisticRegression:
    def __init__(self):
        self.data_n = None
        self.p = None
        self.beta = None

        pass


    #Recieves: Training data (the X and the y), MUST BE MATRICES
    #Does: Calculates the estimated values of beta with IRLS
    #Returns: Nothing
    def fit(self,X,y):
        thres = 0.001
        #print(y.shape)
        #return 

        self.data_n = X.shape[0]
        self.p = X.shape[1]
        self.beta = np.zeros((self.p+1,1))
        #self.beta = self.beta[:,np.newaxis]

        X_design = np.empty((self.data_n,X.shape[1]+1))
        X_design[:,0] = 1.0
        X_design[:,1:] = X
        X_T = np.transpose(X_design)
        p_vec = 1.0 / (1.0 + np.exp(-X_design @ self.beta))  
        #print(X_T.shape)
        #print(W.shape)
        #print(p_vec.shape)
        #print(p_vec)
        #print((X_T @ W @ X).shape)

        #return 

        while True:
            p_vec = 1.0 / (1.0 + np.exp(-X_design @ self.beta)) 
            variance_col = p_vec * (1 - p_vec)  
            old_beta = self.beta
            self.beta = old_beta + np.linalg.inv(X_T @ (variance_col * X_design)) @ X_T @ (y - p_vec)

            if max(np.abs(old_beta - self.beta)) < thres:
                break

        pass

    #Recieves: Data to predict classification
    #Does: Using the precalculated beta vector, calculates the estimated class
    #Returns: The predicted y value
    def predict(self,X):
        if self.beta is None:
            print("First run the fit function to calculate coefficient estimates.")

        n = X.shape[0]
        X_design = np.empty((n,X.shape[1]+1))
        X_design[:,0] = 1.0
        X_design[:,1:] = X

        p_vec = 1.0 / (1.0 + np.exp(-X_design @ self.beta)) 

        return np.where(p_vec >= 0.5, 1.0,0.0)
    
    #Recieves: Nothing
    #Does: Prints out a summary of significant statistics
    #Returns: Nothing
    def summary(self):

        pass

def main():
    rng = np.random.default_rng(seed=1)
    p = 3
    n = 10000

    beta = rng.uniform(-0.5, 0.5, size=(p+1, 1))
    
    X = rng.normal(0.0, 1.0, size=(n, p))
    X_design = np.empty((n,p+1))
    X_design[:, 0] = 1.0
    X_design[:, 1:] = X  

    z = X_design @ beta
    probabilities = 1.0 / (1.0 + np.exp(-z))
    y = np.where(rng.random((n, 1)) < probabilities, 1.0, 0.0)

    log_regressor = LogisticRegression()
    log_regressor.fit(X,y)

    y_hat = log_regressor.predict(X)

    print(beta.T)
    print(log_regressor.beta.T)





    pass 

if __name__ == "__main__":
    main()
