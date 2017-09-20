# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 22:37:16 2017

@author: Matteo

"""
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.signal import detrend
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, AveragePooling1D
from keras.layers import Flatten
from keras.utils import np_utils
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
import keras.backend as K
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
#%% import data
data_Ts_pd = pd.read_csv(r'D:\dataset_SAFE_slide\dataset_SAFE_slide\dati\intel_Ts.csv',header = None)
data_X_pd = pd.read_csv(r'D:\dataset_SAFE_slide\dataset_SAFE_slide\dati\intel_X.csv',header = None)
data_Y_pd = pd.read_csv(r'D:\dataset_SAFE_slide\dataset_SAFE_slide\dati\intel_Y.csv',header = None)

#%%
data_Ts = data_Ts_pd.as_matrix()
data_X = data_X_pd.as_matrix()
data_Y = np.array(data_Y_pd)

#%%

data_Ts = np.reshape(data_Ts, (1771,54))
data_X = np.reshape(data_X,(1771,54,2048))

idx = data_Y > 60
data_Y = data_Y[idx]
y_offset = 70
data_Y = data_Y - y_offset
data = data_X[:-sum(np.logical_not(idx))[0]]

#%%
data_X_1 = np.reshape(data,(1747,54*2048))
#k = 100
#f = SelectKBest(f_regression, k=k)
#X_new = f.fit_transform(data_X_1, np.repeat(data_Y,54))

k = 100
#PCA
pca = PCA(n_components = k)
data_X = pca.fit_transform(data_X_1) 

##%%
#n_jobs = 6
#forest = ExtraTreesRegressor(n_estimators=100,
#                              random_state=0,
#                              max_features=150,
#                              n_jobs = n_jobs)
#
#forest.fit(data_X_1, np.repeat(data_Y,54)[0:54*800])
#importances = forest.feature_importances_
#std = np.std([forest.feature_importances_ for tree in forest.estimators_],
#             axis=0)
#indices = np.argsort(importances)[::-1]
#plt.bar(range(data_X_1.shape[1]), importances[indices],
#       color="r", yerr=std[indices], align="center")
#plt.show()

#%%
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.25, random_state=101)
X_train = normalize(X_train, norm = 'max')
X_test = normalize(X_test, norm = 'max')

# %% DENSE Model

batch_size = 32
epochs = 30
n_dense_1 = 50
n_dense_2 = 50
n_dense_3 = 30
n_dense_4 = 30
#n_dense_5 = 30


model = Sequential()
model.add(Dense(n_dense_1, 
                activation='relu',
                input_shape = (k,)))
model.add(Dense(n_dense_2, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(n_dense_3, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(n_dense_4, activation='relu'))
model.add(Dropout(0.1))
#model.add(Dense(n_dense_5, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(1, kernel_initializer='normal',
                use_bias = True,
                activation = None))	
model.summary()


#%%
# Compile model	
def r_score_metric(y_true, y_pred):
    y_true = K.tensorflow_backend._to_tensor(y_true, dtype = tf.float32)
    y_pred = K.tensorflow_backend._to_tensor(y_pred, dtype = tf.float32)
#    return  1 - K.sum(y_true - y_pred)
    return  1 - K.sum((y_true - y_pred)**2)/(K.sum((y_true - K.mean(y_true))**2))

model.compile(loss=['mean_squared_error'],
              metrics = ['mean_squared_error', r_score_metric],
              optimizer='adam')
K.get_session().run(tf.global_variables_initializer())
model.fit(X_train, y_train,  
          batch_size=batch_size,      
          epochs=epochs,          
          verbose=1,          
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('R: ', score[-1])

y_pred = model.predict(X_test)
y_pred = y_pred[:,0]
#gof = 1 - np.linalg.norm(y_pred - y_test)/np.linalg.norm(y_test - np.mean(y_test))
r_score = r2_score(y_test, y_pred)

#print('R: ', r_score)


#%%

plt.figure()
plt.scatter(y_pred,y_test)



#%%

#X_train_1 = normalize(X_train[:,:,0], norm = 'max')
#X_test_1 = normalize(X_test[:,:,0], norm = 'max')

svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})
   #svr_poly = SVR(kernel='poly', C=1e3, degree=2)

model_rbf = svr.fit(X_train_1, y_train)
#model_poly = svr_poly.fit(X_train, y_train)
y_rbf = model_rbf.predict(X_test_1)
score_rbf = model_rbf.score(X_test_1,y_test)

plt.figure()
plt.scatter(y_rbf,y_test)
