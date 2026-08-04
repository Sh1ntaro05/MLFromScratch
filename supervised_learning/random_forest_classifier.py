import numpy as np
import pandas as pd
from supervised_learning.decision_tree_classifier import DecisionTreeClassifier, Node
from utilities.datasplit import data_split

class RandomForestClassifier:
    def __init__(self):
        self.tree_roots = None
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
        self.m = (int) (np.sqrt(self.p))
        self.tree_roots = np.empty(num_trees,dtype=object)
        self.features = [i for i in range(self.p)]
        self.data_ids = [i for i in range(self.n)]
        self.num_trees = num_trees
        whole_data = np.hstack((X,y))
        self.rng = np.random.default_rng(seed)
        if(criterion != 'entropy' and criterion != 'gini'):
            print("Criterion must be entropy or gini")
            return
        
        for i in range(num_trees):
            bootstraped_data = self.bootstrap(whole_data)
            self.tree_roots[i] = Node(data=bootstraped_data,depth=1)
            if(criterion == 'entropy'):
                self.tree_roots[i].score = self.get_entropy(self.tree_roots[i].data[:,-1])
                self.tree_roots[i].left,self.tree_roots[i].right = self.build_tree(self.tree_roots[i],max_depth,min_samples_split,self.get_entropy)
            elif(criterion == 'gini'):
                self.tree_roots[i].score = self.get_gini(self.tree_roots[i].data[:,-1])
                self.tree_roots[i].left,self.tree_roots[i].right = self.build_tree(self.tree_roots[i],max_depth,min_samples_split,self.get_gini)
        return


    #Recieves: Data to bootstrap from
    #Does: Get n samples from the data with replacement
    #Returns: The bootstrapped data sample
    def bootstrap(self, all_data):
        data_ids_use = self.rng.choice(self.data_ids,size=self.n,replace=True)

        bootstrap_sample = all_data[data_ids_use]

        return bootstrap_sample

    #Recieves: Current node, maxdepth, and the score function to use(gini or entropy)
    #Does: Recursively builds the decision tree
    #Returns: A tuple: (leftnode, rightnode) (Null node if terminal conditions are met)
    def build_tree(self,curr_node,max_depth,min_samples_split,score_func):
        if curr_node.score == 0.0 or curr_node.depth == max_depth:
            curr_node.value = self.get_majority_vote(curr_node.data[:,-1])
            curr_node.data = None 
            return (None,None) 
        
        left_data, right_data, left_score, right_score, feature, thres = self.find_best_condition(curr_node.data,score_func)

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


    #Recieves: Data to find the best split on(the right-most column of the data are the labels)
    #and the score function to use(gini or entropy)
    #Does: Traverses through possible conditions and finds the best one(only looks at the randomly selected features)
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
        chosen_features = self.rng.choice(self.features,size=self.m,replace=False)
        for feature in chosen_features:
            thres_candidates = np.unique(data[:,feature])
            for thres in thres_candidates:
                left_mask = data[:,feature] <= thres
                right_mask = data[:,feature] > thres
                left_data = data[left_mask]
                right_data = data[right_mask]
                
                left_weight = len(left_data) / data_num
                right_weight = len(right_data) / data_num
                left_score = score_func(left_data[:,-1])
                right_score = score_func(right_data[:,-1])

                gain = curr_score - left_weight * left_score - right_weight * right_score
                if(gain > max_gain):
                    max_gain = gain 
                    best_feature = feature 
                    best_thres = thres
                    best_left = left_data
                    best_right = right_data
                    best_left_score = left_score 
                    best_right_score = right_score 

        return (best_left, best_right, best_left_score, best_right_score, best_feature, best_thres)
    
    #Recieves: Data to predict the label
    #Does: Labels the test data using the preconstructed random forest
    #Returns: An array of predicted labels 
    def predict(self, X):
        n = X.shape[0]

        predictions = np.empty((n,1), dtype=object)
        votes = np.empty((self.num_trees,1),dtype=object)

        for i in range(n):
            for j in range(self.num_trees):
                votes[j] = self.predict_row(self.tree_roots[j],X[i])
            predictions[i] = self.get_majority_vote(votes)

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

    #Recieves: The label portion of a subset of the original data
    #Does: Calculates the majority vote of that subset
    #Returns: The majority vote(if there are multiple, return a random majority label)
    def get_majority_vote(self,labels):
        values, counts = np.unique(labels, return_counts=True)
        mask = (counts == max(counts)) 

        if np.sum(mask) == 1:
            return values[mask][0]
        
        return self.rng.choice(values[mask])
    
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









