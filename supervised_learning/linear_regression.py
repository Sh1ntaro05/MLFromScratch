import numpy as np

class LinearRegression:
    def __init__(self):
        self.beta = None 

    #Recieves: Training data (the X and the y)
    #Does: Calculates important statistics including the beta, variance of beta, etc.
    #Returns: Nothing
    def fit(self,X,y):
        n = X.shape[0]
        X_design = np.empty((n,X.shape[1]+1))
        X_design[:,0] = 1.0
        X_design[:,1:] = X
        X_T = np.transpose(X_design)
        self.beta = np.linalg.inv(X_T @ X_design) @ (X_T @ y)

    #Recieves: Data to predict y value
    #Does: Using the precalculated beta vector, calculates the estimation of y
    #Returns: The predicted y value
    def predict(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1,-1)

        n = X.shape[0]
        X_design = np.empty((n,X.shape[1]+1))
        X_design[:,0] = 1.0
        X_design[:,1:] = X

        return X_design @ self.beta
    
    def summary(self):
        pass


def main():
    rng = np.random.default_rng(seed=1)
    p = 5
    n = 10

    beta = 10 * rng.random((p+1,1))
    
    X = 10 * rng.random((n,p))
    X_design = np.empty((n,p+1))
    X_design[:, 0] = 1.0
    X_design[:, 1:] = X    

    noise = rng.normal(0,1.0,(n,1))

    y = X_design @ beta + noise

    lin_regressor = LinearRegression()
    lin_regressor.fit(X,y)

    beta_hat = lin_regressor.beta
    print(beta.T)
    print(beta_hat.T)


    pass

if __name__ == "__main__":
    main()

