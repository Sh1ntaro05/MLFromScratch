import numpy as np
import pandas as pd
from utilities.datasplit import data_split

class QDAClassifer:
    def __init__(self):
        self.n = None 
        self.p = None 
        self.k = None
        self.labels = None
        self.info_k = {}        
        pass 

    def fit(self,X,y):
        y = y.flatten()
        self.n = X.shape[0]
        self.p = X.shape[1]  
        self.labels, counts = np.unique(y, return_counts=True)
        #print(self.labels,counts)
        for label,count in zip(self.labels,counts):
            #print(label)
            self.info_k[label] = {}
            #self.info_k[label]["label"] = label
            self.info_k[label]["prior"] = count / self.n 
            data_k = self.find_data(label,X,y)
            self.info_k[label]["mean_vec"] = np.reshape(np.mean(data_k,axis=0),(-1,1))
            self.info_k[label]["var_matrix"] = np.cov(data_k,rowvar=False)
            self.info_k[label]["var_matrix_det"] = np.linalg.det(self.info_k[label]["var_matrix"])
            if self.info_k[label]["var_matrix_det"] == 0:
                #print(np.shape(self.info_k[label]["var_matrix"]))
                alpha = 1e-5
                I_p = np.identity(self.p)
                self.info_k[label]["var_matrix"] += alpha * I_p
                self.info_k[label]["var_matrix_det"] = np.linalg.det(self.info_k[label]["var_matrix"])
            self.info_k[label]["var_matrix_inv"] = np.linalg.inv(self.info_k[label]["var_matrix"])

        self.k = len(self.labels)


        pass 

    #Revieves: Data to predict that class
    #Does: For each row, calculates the most likely class using the helper functions
    #Returns: An array of the estimated labels for each row
    def predict(self,X):
        return np.array([self.find_max_class(row) for row in X])

    #Recieves: A class and one row of data
    #Does: Calculates the probability P(Y=k|X=x)
    #Returns: The calculated probability
    def delta_k(self,label,x):
        x = np.reshape(x,(-1,1))
        mu_k  = self.info_k[label]["mean_vec"]
        sigma_k_det = self.info_k[label]["var_matrix_det"]
        sigma_k_inv = self.info_k[label]["var_matrix_inv"] 
        pi_k = self.info_k[label]["prior"]
        return -0.5 * ((x - mu_k).T @ sigma_k_inv) @ (x - mu_k) - 0.5 * np.log(sigma_k_det) + np.log(pi_k)

    #Recieves: A class and the X and y data
    #Does: Finds the rows in X that correspond to the given label
    #Returns: A matrix that is a part of X whose class is the specified label
    def find_data(self, label, X, y):
        mask = (y == label)
        #print(np.shape(mask))
        #print(mask)
        #print(X[mask])
        return X[mask]
    
    #Recieves: A row to find the class
    #Does: Finds the class that maximises the probability P(Y=k|X=x)
    #Returns: The class determined by the process
    def find_max_class(self, x):
        #print(x)
        max_prob = -np.inf
        max_label = self.labels[0]
        for label in self.labels:
            #print(self.delta_k(label,x))
            if self.delta_k(label,x) > max_prob:
                max_prob = self.delta_k(label,x)
                max_label = label 
        return max_label




def main():
    iris_df = pd.read_csv('./assets/iris_dataset/Iris.csv')

    X = iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
    X = X.to_numpy()
    y = iris_df[['Species']]
    y = y.to_numpy()
    X_train,X_test,y_train,y_test = data_split(X,y)

    QDA = QDAClassifer()
    QDA.fit(X_train,y_train)

    y_hat = QDA.predict(X_test)
    #print(np.shape(y_test),np.shape(y_hat))
    print(f"Accuracy: {np.sum(y_hat == y_test.flatten()) / y_test.shape[0]}")


    pass 

if __name__ == "__main__":
    main()
