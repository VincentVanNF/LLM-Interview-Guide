import pandas as pd
import scipy as sp
import numpy as np
import json
from sklearn.cluster import DBSCAN
import sklearn
#基于DBSCAN异常检测
'''
{
    "train": [[0,0],[0.1,0],[5,5],[5.1,5],[10,0]], 
    "test": [[0.05,0.05],[5.05,5.05],[9,0]]
}
'''
data = {
    "train": [[0,0],[0.1,0],[5,5],[5.1,5],[10,0]], 
    "test": [[0.05,0.05],[5.05,5.05],[9,0]]
}
X_train = np.array(data['train'])
X_test = np.array(data['test'])

x_all = np.vstack((X_train,X_test))
x_all = sklearn.preprocessing.StandardScaler().fit_transform(x_all)

cluster = DBSCAN(
    eps=0.3, 
    min_samples=3,
    metric = "euclidean",
    algorithm='auto'
)
res = cluster.fit(x_all)
res = res.labels_[-len(X_test):]
print(type(res))