# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:48:58 2017

@author: Matteo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:00:59 2016

@author: chiara

Train recurrent convolutional network 
"""
#from __future__ import print_function
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
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv1D, AveragePooling1D, BatchNormalization
from keras.layers import Flatten
from keras.utils import np_utils


#%% import data
#data_Ts_pd = pd.read_csv(r'D:\dataset_SAFE_slide\dataset_SAFE_slide\dati\intel_Ts.csv',header = None)
#data_X_pd = pd.read_csv(r'D:\dataset_SAFE_slide\dataset_SAFE_slide\dati\intel_X.csv',header = None)
#data_Y_pd = pd.read_csv(r'D:\dataset_SAFE_slide\dataset_SAFE_slide\dati\intel_Y.csv',header = None)
#
##%%
#
#
#data_Ts = data_Ts_pd.as_matrix()
#data_X = data_X_pd.as_matrix()
#data_Y = np.array(data_Y_pd)
#
#data_Ts = np.reshape(data_Ts, (1771,54))
#data_X = np.reshape(data_X,(1771,54,2048))

data_Ts_pd = pd.read_csv(r'C:\Users\RDPuser\Documents\Data\intel_Ts.csv',header = None)
data_X_pd = pd.read_csv(r'C:\Users\RDPuser\Documents\Data\intel_X.csv',header = None)
data_Y_pd = pd.read_csv(r'C:\Users\RDPuser\Documents\Data\intel_Y.csv',header = None)
#%%

# -----------------MIA PROCEDURA
data_Ts = data_Ts_pd.as_matrix()
data_X = data_X_pd.as_matrix()
data_Y = np.array(data_Y_pd)

data_Ts = np.reshape(data_Ts, (1771,54))
data_X = np.reshape(data_X,(1771,54,2048))
#data_X = np.reshape(data_X, (data_X.shape[0],1,54,2048))
#
#
idx = data_Y > 60
data_Y = data_Y[idx]
data_Y = data_Y - 70
idx = np.reshape(idx, (1771))
data_X = data_X[idx]
#
#data_X = data_X[:-np.sum(np.logical_not(idx))]
#data_Y = data_Y[:-np.sum(np.logical_not(idx))]
#X_mean = np.mean(data_X, axis=(1,2))
#coeff = np.corrcoef(X_mean, data_Y)
#
#plt.scatter(X_mean, data_Y)

#data_mean = np.array(list(map(lambda z: np.apply_along_axis(lambda x: np.mean(x),0,z),data_X)))

#data_X = data_X.swapaxes(2,1)
        
#data_X = np.reshape(data_X, (data_X.shape[0],1,54,2048))

#------------------PROCEDURA MATTEO
#data_Ts = data_Ts_pd.as_matrix()
#data_X = data_X_pd.as_matrix()
#data_Y = np.array(data_Y_pd)
#
#data_Ts = np.reshape(data_Ts, (1771,54))
#data_X = np.reshape(data_X,(1771,54,2048))
#
#idx = data_Y > 60
#data_Y = data_Y[idx]
#y_offset = 70
#data_Y = data_Y - y_offset
#plt.figure()
#data_X = data_X[:-sum(np.logical_not(idx))[0]]
#X_mean1=np.mean(data_X, axis=(1,2))
#coeff1 = np.corrcoef(X_mean1, data_Y)
#
#plt.figure()
#plt.scatter(X_mean1, data_Y)
#
#data_XX = np.reshape(data_X,(len(data_X),54,2048))
#
#plt.figure()
#plt.plot(data_Y[data_Y > 65]*8, 'r')
#plt.plot(np.mean(np.mean(data_XX, axis = 1), axis = 1))
#plt.show()
#%%
# split into 67% for train and 33% for test
data_X = data_X.swapaxes(2,1)
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.25, random_state=101)


# %% CNN Model
# CNN model for regression
filter_depth = 8
filter_depth2 = 4
filter_depth3 = 2
a = 8
a2 = 4
a3 = 2
am = 8
am2 = 4
am3 = 2    
batch_size = 32
epochs = 30
n_dense_1 = 10
n_dense_2 = 50
#n_dense_3 = 10


model = Sequential()
model.add(Conv1D(filter_depth,
                 kernel_size= a,
                 activation='relu', 
                 strides=4, 
                 padding='same', 
                 use_bias=True, 
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros', 
#                 kernel_regularizer=None, 
#                 bias_regularizer=None, 
#                 activity_regularizer=None, 
#                 kernel_constraint=None, 
#                 bias_constraint=None,
                 input_shape=(2048,54)))
#model.add(BatchNormalization(axis = -1))
model.add(AveragePooling1D(pool_size=am))
model.add(Conv1D(filter_depth2,
                 kernel_size=(a2),
                 activation='relu', 
                 strides=2,
                 padding='same', 
                 use_bias=True, 
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros', 
#                 kernel_regularizer=None, 
#                 bias_regularizer=None, 
#                 activity_regularizer=None, 
#                 kernel_constraint=None, 
#                 bias_constraint=None
))
#model.add(BatchNormalization(axis = -1))
model.add(AveragePooling1D(pool_size=am2))
model.add(Conv1D(filter_depth3,
                 kernel_size=a3,
                 activation='relu', 
                 strides=1, 
                 padding='same', 
                 use_bias=True, 
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros', 
#                 kernel_regularizer=None, 
#                 bias_regularizer=None, 
#                 activity_regularizer=None, 
#                 kernel_constraint=None, 
#                 bias_constraint=None
))
#model.add(BatchNormalization(axis = -1))
model.add(AveragePooling1D(pool_size=am3))
model.add(Flatten())
#model.add(Dropout(0.1))
model.add(Dense(n_dense_1, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, kernel_initializer='normal',
                use_bias = True,
                activation=None))	
model.summary()


#%%
# Compile model	
model.compile(loss=['mean_squared_error'],
              metrics = ['mean_squared_error','mean_absolute_percentage_error'],
              optimizer='adam')

#%%fit
model.fit(X_train, y_train,  
          batch_size=batch_size,      
          epochs=epochs,          
          verbose=1,          
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)

y_pred = model.predict(X_test)
#gof = 1 - np.linalg.norm(y_pred - y_test)/np.linalg.norm(y_test - np.mean(y_test))
r_score = r2_score(y_test, y_pred)

print('R: ', r_score)


#%%

plt.figure()
plt.scatter(y_pred,y_test)
