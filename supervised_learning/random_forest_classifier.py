import numpy as np
import pandas as pd
from supervised_learning.decision_tree_classifier import DecisionTreeClassifier, Node
from utilities.datasplit import data_split

class RandomForestClassifier:
    def __init__(self):
        self.trees = None 
        self.n = None 
        self.p = None 
        self.m = None
        self.features = None 
        self.data_ids = None 
        self.num_trees = None 
        self.rng = None 

    def fit(self,X,y,num_trees,criterion='entropy',max_depth=2,min_samples_split=2,seed=1):
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.m = int(np.sqrt(self.p))
        self.trees = [None for _ in range(num_trees)]
        self.features = [i for i in range(self.p)]
        self.data_ids = [i for i in range(self.n)]
        self.num_trees = num_trees
        self.rng = np.random.default_rng(seed)
        whole_data = np.hstack((X,y))
        if(criterion != 'entropy' and criterion != 'gini'):
            print("Criterion must be entropy or gini")
            return

        for i in range(num_trees):
            bootstrap_sample = self.bootstrap(whole_data)
            self.trees[i] = DecisionTreeClassifier()
            self.trees[i].fit(bootstrap_sample[:,:-1],bootstrap_sample[:,-1:],criterion,max_depth,min_samples_split,seed,self.m)
            seed += 1

        return

    #Recieves: Data to conduct prediction on
    #Does: For each row in X, pass it to all of the individual trees and find the majority vote
    #Returns: Which class each row in X likely belongs to
    def predict(self,X):
        n = X.shape[0]

        all_tree_preds = np.empty((self.num_trees,n),dtype=object)

        for i in range(self.num_trees):
            all_tree_preds[i] = self.trees[i].predict(X).squeeze()

        ensemble_preds = np.empty((n,1),dtype=object)

        predictions = [None for _ in range(self.num_trees)]

        for j in range(n):
            values, counts = np.unique(all_tree_preds[:,j],return_counts=True)
            ensemble_preds[j] = values[np.argmax(counts)]

        return ensemble_preds

    #Recieves: Data to conduct prediction on and the class to measure probability
    #Does: For each row in X, calculates the probability that the data point belongs in target_class
    #Returns: The probability that each row belongs in target_class
    def predict_proba(self,X,target_class):
        n = X.shape[0]

        all_tree_preds = np.empty((self.num_trees,n),dtype=object)

        for i in range(self.num_trees):
            all_tree_preds[i] = self.trees[i].predict(X).squeeze()

        probabilities = np.zeros((n,1))

        for j in range(n):
            votes = np.sum(all_tree_preds[:,j] == target_class)
            probabilities[j] = votes / self.num_trees

        return probabilities    
            
    #Recieves: Data to bootstrap from
    #Does: Get n samples from the data with replacement
    #Returns: The bootstrapped data sample
    def bootstrap(self, whole_data):
        data_ids_use = self.rng.choice(self.data_ids,size=self.n,replace=True)
        bootstrap_sample = whole_data[data_ids_use]
        return bootstrap_sample


#Recieves: The root node
#Does: Resursively traverses the tree and prints attributes
#Returns: Nothing
def printTree(node):
    if node != None:
        printTree(node.left)
        print(f"depth: {node.depth}, feature: {node.feature}, threshold: {node.thres}, score: {node.score} \n")
        printTree(node.right)


def main():
    iris_df = pd.read_csv('./assets/iris_dataset/Iris.csv')

    X = iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
    X = X.to_numpy()
    y = iris_df[['Species']]
    y = y.to_numpy()
    X_train,X_test,y_train,y_test = data_split(X,y)

    RFC = RandomForestClassifier()
    RFC.fit(X_train,y_train,num_trees=1000,criterion='gini',max_depth=100,seed=1)

    y_hat = RFC.predict(X_test)
    print(f"Accuracy: {np.sum(y_hat == y_test) / y_test.shape[0]}")
    
    pass 

if __name__ == "__main__":
    main()









