import numpy as np
import pandas as pd
from utilities.datasplit import data_split

class Node:
    def __init__(self,left=None,right=None,feature=None,thres=None,data=None,score=None,depth=None):
        self.left = left
        self.right = right
        self.feature = feature
        self.thres = thres
        self.data = data
        self.score = score
        self.depth = depth
        self.value = None
        pass

class DecisionTreeClassifier:
    def __init__(self):
        self.root = None
        self.n = None
        self.p = None
        #self.K = None
        self.rng = None
        self.max_features = None

        pass

    def fit(self,X,y,criterion='entropy',max_depth=2,min_samples_split=2,seed=1,max_features=None):
        self.n = X.shape[0]
        self.p = X.shape[1]
        #self.classes = np.unique(y)
        #self.K = len(self.classes)
        self.rng = np.random.default_rng(seed)
        
        self.max_features = max_features if max_features is not None else self.p
        whole_data = np.hstack((X,y))
        self.root = Node(data=whole_data,depth=1)
        if(criterion != 'entropy' and criterion != 'gini'):
            print("Criterion must be entropy or gini")
            return
        

        if(criterion == 'entropy'):
            self.root.score = self.get_entropy(self.root.data[:,-1])
            self.root.left,self.root.right = self.build_tree(self.root,max_depth,min_samples_split,self.get_entropy)
        elif(criterion == 'gini'):
            self.root.score = self.get_gini(self.root.data[:,-1])
            self.root.left,self.root.right = self.build_tree(self.root,max_depth,min_samples_split,self.get_gini)

        pass

    #Recieves: Current node, maxdepth, and the score function to use(gini or entropy)
    #Does: Recursively builds the decision tree
    #Returns: A tuple: (leftnode, rightnode) (Null node if terminal conditions are met)
    def build_tree(self,curr_node,max_depth,min_samples_split,score_func):
        if curr_node.score < 1e-9 or curr_node.depth == max_depth or len(curr_node.data) < min_samples_split:
            curr_node.value = self.get_majority_vote(curr_node.data[:,-1])
            curr_node.data = None
            return (None,None)
        
        left_data,right_data,left_score,right_score,feature,thres = self.find_best_condition(curr_node.data,score_func)
        
        if feature is None:
            curr_node.value = self.get_majority_vote(curr_node.data[:,-1])
            curr_node.data = None
            return (None,None)

        left_node = Node(data=left_data,score=left_score,depth=curr_node.depth+1)
        right_node = Node(data=right_data,score=right_score,depth=curr_node.depth+1)
        
        curr_node.feature = feature
        curr_node.thres = thres
        curr_node.data = None
        
        left_node.left,left_node.right = self.build_tree(left_node,max_depth,min_samples_split,score_func)
        right_node.left,right_node.right = self.build_tree(right_node,max_depth,min_samples_split,score_func)
        return (left_node,right_node)
    
    #Recieves: The label portion of a subset of the original data
    #Does: Calculates the majority vote of that subset
    #Returns: The majority vote(if there are multiple, return a random majority label)
    def get_majority_vote(self,labels):
        values, counts = np.unique(labels, return_counts=True)
        mask = (counts == max(counts)) 

        if np.sum(mask) == 1:
            return values[mask][0]
        
        rng = np.random.default_rng()
        return rng.choice(values[mask])

    #Recieves: Data to predict the label
    #Does: Labels the test data using the preconstructed DTC
    #Returns: An array of predicted labels 
    def predict(self, X):
        n = X.shape[0]

        predictions = np.empty((n,1), dtype=object)

        for i in range(n):
            predictions[i] = self.predict_row(self.root, X[i])
        
        return predictions
    
    #Recieves: One row to predict label
    #Does: Goes through the preconstructed DTC recursively and returns label
    #Returns: Label
    def predict_row(self, node, row):
        if node.value != None:
            return node.value

        if row[node.feature] <= node.thres:
            return self.predict_row(node.left, row)
        else:
            return self.predict_row(node.right, row)

    #Recieves: The label portion of a subset of the original data
    #Does: Calculates the gini index of the subset
    #Returns: The gini index
    def get_gini(self,labels):
        num_labels = len(labels)
        _, counts = np.unique(labels, return_counts=True)
        probabilties = counts / num_labels
        return np.sum(probabilties * (1-probabilties))

    #Recieves: The label portion of a subset of the original data
    #Does: Calculates the entropy of the subset
    #Returns: The entropy
    def get_entropy(self,labels):
        num_labels = len(labels)
        _, counts = np.unique(labels, return_counts=True)
        probabilties = counts / num_labels
        return np.sum(-probabilties * np.log2(probabilties))

    #Recieves: Data to find the best split on(the right-most column of the data are the labels) and the score function to use(gini or entropy)
    #Does: Traverses through possible conditions and finds the best one
    #Returns: A tuple: (leftdata, rightdata, leftscore, rightscore, feature#, threshold)
    def find_best_condition(self,data,score_func):
        curr_score = score_func(data[:,-1])
        max_gain = 0.0
        data_num = len(data)
        best_feature = None
        best_thres = None
        best_left = None
        best_right = None
        best_left_score = None
        best_right_score = None

        chosen_features = self.rng.choice(self.p,size=self.max_features,replace=False)

        for i in chosen_features:
            thres_candidates = np.unique(data[:,i])
            for thres in thres_candidates:
                left_mask = data[:,i] <= thres
                right_mask = data[:,i] > thres
                left_data = data[left_mask]
                right_data = data[right_mask]

                if len(left_data) == 0 or len(right_data) == 0:
                    continue
                
                left_weight = len(left_data) / data_num
                right_weight = len(right_data) / data_num
                left_score = score_func(left_data[:,-1])
                right_score = score_func(right_data[:,-1])

                gain = curr_score - left_weight * left_score - right_weight * right_score
                if(gain > max_gain):
                    max_gain = gain 
                    best_feature = i 
                    best_thres = thres
                    best_left = left_data
                    best_right = right_data
                    best_left_score = left_score 
                    best_right_score = right_score 

        return (best_left, best_right, best_left_score, best_right_score, best_feature, best_thres)

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

    DTC = DecisionTreeClassifier()
    DTC.fit(X_train,y_train,criterion='gini',max_depth=2,min_samples_split=3)

    y_hat = DTC.predict(X_test)
    print(f"Accuracy: {np.sum(y_hat == y_test) / y_test.shape[0]}")
    
    pass

if __name__ == "__main__":
    main()
