import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utilities.distance_functions as DistFuncs

class KMeans:
    def __init__(self,num_clusters,max_iters=100,dist_func=DistFuncs.euclid_dist):
        self.dims = None 
        self.n_data = None 
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.dist_func = dist_func
        self.centroids = None 
        self.clusterings = None 
        
        
    
    def fit(self, X):
        thres = 1e-7
        self.dims = X.shape[1]
        self.n_data = X.shape[0]
        WCSS = np.zeros(self.num_clusters)
        init_centroid_idx = np.random.choice(self.n_data,size=self.num_clusters,replace=False)
        self.centroids = X[init_centroid_idx]
        self.clusterings = np.zeros(self.n_data)            

        for _ in range(self.max_iters):
            for i in range(self.n_data):
                self.clusterings[i] = self.find_cluster(X[i],self.centroids)
            old_centroids = self.centroids.copy()
            for i,centroid in enumerate(self.centroids):
                mask = self.clusterings == i
                count = np.sum(mask)
                self.centroids[i] = np.sum(X[mask],axis=0) / count
            if np.max(np.abs(old_centroids-self.centroids)) < thres:
                break 
        
        return self.clusterings

    def predict(self,X):
        pass 

    def find_cluster(self,x,centroids):
        min_dist = np.inf
        cluster_idx = 0
        for i,centroid in enumerate(centroids):
            if self.dist_func(x,centroid) < min_dist:
                min_dist = self.dist_func(x,centroid)
                cluster_idx = i 
        return cluster_idx
            



        
    
def main():
    n = 1000
    K = 5
    p = 2
    X = np.random.randn(n,p)

    Kmeans_model = KMeans(K)
    y = Kmeans_model.fit(X)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.xlabel("1st coordinate")
    plt.ylabel("2nd cooridnate")

    plt.savefig("kmeans_clusters.png", dpi=300, bbox_inches="tight")


    pass 

if __name__ == "__main__":
    main()
