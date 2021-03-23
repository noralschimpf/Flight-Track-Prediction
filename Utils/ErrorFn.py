import numpy as np

# Distance Error Reports: Mean Average  Pointwise Error: HORIZONTAL, Along-Track, Cross-Track, VERTICAL



# Distance Error Reports: L2 (Euclidean) Norm, percent reduction


def L2Norm(true: np.array, test: np.array):
    if true.shape != test.shape:
        raise ValueError('True and Test array shapes must match dimenions')
    norm = 0
    for i in range(true.shape[0]):
        dist  = 0
        for j in range(true.shape[1]):
            dist += abs((true[i,j] - test[i,j]))**2
        norm += np.sqrt(dist)
    return np.sqrt(norm)

def MSE(act: np.array, test:np.array):
    if act.shape != test.shape:
        raise ValueError('True and Test array shapes must match dimenions')
    mse = 0
    # for each point
    for i in range(act.shape[0]):
        # for each dimension
        dist = 0
        for j in range(act.shape[1]):
            dist += abs((act[i,j] - test[i,j]))**2
        mse += np.sqrt(dist)
    return mse/act.shape[0]

def reduction(l2_orig, l2_new):
    if isinstance(l2_orig, np.ndarray) or isinstance(l2_orig, list):
        if len(l2_orig) >1:
            return (np.var(l2_orig) - np.var(l2_new))/np.var(l2_orig)
    else:
        return (l2_orig - l2_new)/l2_orig