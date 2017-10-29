# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:06:16 2017

@author: Matteo
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
seed = 1337
np.random.seed(seed)  # for reproducibility

#%% import data
#data_Ts_pd = pd.read_csv(r'D:\dataset_SAFE_slide\dataset_SAFE_slide\dati\intel_Ts.csv',header = None)
#data_X_pd = pd.read_csv(r'D:\dataset_SAFE_slide\dataset_SAFE_slide\dati\intel_X.csv',header = None)
#data_Y_pd = pd.read_csv(r'D:\dataset_SAFE_slide\dataset_SAFE_slide\dati\intel_Y.csv',header = None)

data_Ts_pd = pd.read_csv(r'C:\Users\RDPuser\Documents\Data\intel_Ts.csv',header = None)
data_X_pd = pd.read_csv(r'C:\Users\RDPuser\Documents\Data\intel_X.csv',header = None)
data_Y_pd = pd.read_csv(r'C:\Users\RDPuser\Documents\Data\intel_Y.csv',header = None)
#%%


data_Ts = data_Ts_pd.as_matrix()
data_X = data_X_pd.as_matrix()
data_Y = np.array(data_Y_pd)

data_Ts = np.reshape(data_Ts, (1771,54))
data_X = np.reshape(data_X,(1771,54,2048))
#data_X = np.reshape(data_X, (data_X.shape[0],1,54,2048))


idx = data_Y > 60
data_Y = data_Y[idx]
idx = np.reshape(idx, (1771))
data_X = data_X[idx]
data_mean = np.array(list(map(lambda z: np.apply_along_axis(lambda x: np.mean(x),0,z),data_X)))
#idx_rand = np.random.randint(0,2047,size = 100)
#data_mean2 = data_mean[:,idx_rand]

y_offset = 70
data_Y = data_Y - y_offset


#%%

#PCA
k = 100
pca = PCA(n_components=k)
X_new = pca.fit_transform(data_mean) 

data_X = np.reshape(X_new,(1747,k))
#%%
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.25, random_state=101)

#%%

svr_rbf = SVR(kernel='rbf', C=0.01, gamma=0.1)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)

model_rbf = svr_rbf.fit(X_train, y_train)
#model_poly = svr_poly.fit(X_train, y_train)
y_rbf = model_rbf.predict(X_test)
#y_poly = model_poly.predict(X_test)

r_score_rbf = r2_score(y_test, y_rbf)
print(r_score_rbf)

plt.plot(y_rbf)
plt.plot(y_test)

#%%


model = Ridge(alpha = 1)
seed = 1337
np.random.seed(seed)

n_splits = 10
frac = 1.0/n_splits*(n_splits - 1)
N = len(data_X)
score_vec = np.zeros((n_splits,1))
#it = 0
#for i in range(n_splits):
#    idx = np.random.permutation(N)
#    train_idx = idx[:int(frac*N)]
#    test_idx = idx[int(frac*N) + 1:]
#    X_train = data_X[train_idx]
#    y_train = data_Y[train_idx]
#    X_test = data_X[test_idx]
#    y_test = data_Y[test_idx]
#    model.fit(X_train, y_train)
#    score = model.score(X_test, y_test)
#    score_vec[i] = score
#    print('R: ', score)
#
#print(score_vec.mean())

idx = np.random.permutation(N)
idx_ar = np.array(np.array_split(idx,n_splits))
l = np.arange(0,n_splits)
for i in range(n_splits):
    train_idx = idx_ar[l[(l < i) | (l > i)]]
    train_idx = np.hstack(train_idx)
    test_idx = idx_ar[l[i]]
    test_idx = np.hstack(test_idx)
    X_train = data_X[train_idx]
    y_train = data_Y[train_idx]
    X_test = data_X[test_idx]
    y_test = data_Y[test_idx]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    score_vec[i] = score
    print('R: ', score)

print(score_vec.mean())


