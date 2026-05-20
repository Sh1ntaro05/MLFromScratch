import numpy as np

def data_split(X,y,test_size=0.2,seed=1):
    N = X.shape[0]
    rng = np.random.default_rng(seed=seed)
    indices = np.arange(N)
    rng.shuffle(indices)
    test_count =int(N*test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return X[train_indices],X[test_indices],y[train_indices],y[test_indices]
    