import numpy as np
from scipy import stats

class LinearRegression:
    def __init__(self):
        self.data_n = None
        self.p = None
        self.beta = None 
        self.rss = None
        self.tss = None
        self.beta_varmatrix = None
        self.stderr = None
        self.tscores = None 
        self.pvals = None
        
        pass

    #Recieves: Training data (the X and the y)
    #Does: Calculates important statistics including the beta, variance of beta, etc.
    #Returns: Nothing
    def fit(self,X,y):
        self.data_n = X.shape[0]
        self.p = X.shape[1]

        X_design = np.empty((self.data_n,X.shape[1]+1))
        X_design[:,0] = 1.0
        X_design[:,1:] = X
        X_T = np.transpose(X_design)
        X_T_X_inv = np.linalg.inv(X_T @ X_design)

        self.beta = X_T_X_inv @ (X_T @ y)
        self.beta = self.beta.reshape(-1,1)

        y_hat = X_design @ self.beta
        
        self.rss = np.sum((y - y_hat) ** 2)
        self.tss = np.sum((y - np.mean(y)) ** 2)
        df = self.data_n - self.p - 1
        sigma_sq = self.rss / df
        self.beta_varmatrix = X_T_X_inv * sigma_sq
        self.stderr = np.sqrt(np.diagonal(self.beta_varmatrix)).reshape(-1,1)
        self.tscores = self.beta / self.stderr
        self.pvals = stats.t.sf(np.abs(self.tscores),df=df) * 2
        
        pass

    #Recieves: Data to predict y value
    #Does: Using the precalculated beta vector, calculates the estimation of y
    #Returns: The predicted y value
    def predict(self, X):
        if self.beta is None:
            print("First run the fit function to calculate coefficient estimates.")

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1,-1)

        n = X.shape[0]
        X_design = np.empty((n,X.shape[1]+1))
        X_design[:,0] = 1.0
        X_design[:,1:] = X

        return X_design @ self.beta
    
    #Recieves: Nothing
    #Does: Prints out a summary of significant statistics
    #Returns: Nothing
    def summary(self):
        print(f"beta_hat: {self.beta.T}")
        print(f"std. error: {self.stderr.T}")
        print(f"t score: {self.tscores.T}")
        print(f"P values: {self.pvals.T}")

        pass


def main():
    rng = np.random.default_rng(seed=1)
    p = 5
    n = 10
    noise_var = 1.0

    beta = 10 * rng.random((p+1,1))
    
    X = 10 * rng.random((n,p))
    X_design = np.empty((n,p+1))
    X_design[:, 0] = 1.0
    X_design[:, 1:] = X    

    noise = rng.normal(0,noise_var,(n,1))

    y = X_design @ beta + noise

    lin_regressor = LinearRegression()
    lin_regressor.fit(X,y)

    beta_hat = lin_regressor.beta
    print(f"True beta: {beta.T}")
    lin_regressor.summary()

    pass

if __name__ == "__main__":
    main()

