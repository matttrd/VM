#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:00:59 2016

@author: chiara

Train recurrent convolutional network 
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Flatten
from keras.utils import np_utils


#%% import data
data_Ts_pd = pd.read_csv(r'D:\dataset_SAFE_slide\dataset_SAFE_slide\dati\intel_Ts.csv',header = None)
data_X_pd = pd.read_csv(r'D:\dataset_SAFE_slide\dataset_SAFE_slide\dati\intel_X.csv',header = None)
data_Y_pd = pd.read_csv(r'D:\dataset_SAFE_slide\dataset_SAFE_slide\dati\intel_Y.csv',header = None)


#%%


data_Ts = data_Ts_pd.as_matrix()
data_X = data_X_pd.as_matrix()
data_Y = np.array(data_Y_pd)

data_Ts = np.reshape(data_Ts, (1771,54))
data_X = np.reshape(data_X,(1771,54,2048))
data_X = np.reshape(data_X, (data_X.shape[0],1,54,2048))

idx = data_Y > 60
data_Y = data_Y[idx]
data_X = data_X[:-sum(np.logical_not(idx))[0]]

#%%
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.25, random_state=101)

# %% CNN Model
# CNN model for regression
filter_depth = 3
filter_depth2 = 9
#filter_depth3 = 64
a = 5
b = 5*20
a2 = 5
b2 = 5
a3 = 1
b3 = 2
am = 2
bmp = 2
am2 = 2
bmp2 = 2
batch_size = 32
epochs = 30
n_dense_1 = 50
#n_dense_2 = 20
#n_dense_3 = 10


model = Sequential()
model.add(Conv2D(filter_depth,
                 kernel_size=(a, b),
                 activation='relu', 
                 strides=(3, 20), 
                 padding='same', 
                 data_format='channels_first', 
                 use_bias=True, 
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros', 
#                 kernel_regularizer=None, 
#                 bias_regularizer=None, 
#                 activity_regularizer=None, 
#                 kernel_constraint=None, 
#                 bias_constraint=None,
                 input_shape=(1,54,2048)))
model.add(AveragePooling2D(pool_size=(am, bmp),data_format='channels_first'))
model.add(Conv2D(filter_depth2,
                 kernel_size=(a2, b2),
                 activation='relu', 
                 strides=(1, 2), 
                 padding='same', 
                 data_format='channels_first', 
                 use_bias=True, 
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros', 
#                 kernel_regularizer=None, 
#                 bias_regularizer=None, 
#                 activity_regularizer=None, 
#                 kernel_constraint=None, 
#                 bias_constraint=None
))
#model.add(AveragePooling2D(pool_size=(am2, bmp2),data_format='channels_first'))
#model.add(Conv2D(filter_depth3,
#                 kernel_size=(a3, b3),
#                 activation='relu', 
#                 strides=(1, 1), 
#                 padding='same', 
#                 data_format='channels_first', 
#                 use_bias=True, 
#                 kernel_initializer='glorot_uniform', 
#                 bias_initializer='zeros', 
##                 kernel_regularizer=None, 
##                 bias_regularizer=None, 
##                 activity_regularizer=None, 
##                 kernel_constraint=None, 
##                 bias_constraint=None
#))
model.add(Flatten())
#model.add(Dropout(0.1))
model.add(Dense(n_dense_1, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, kernel_initializer='normal',activation=None))	
model.summary()


#%%
# Compile model	
model.compile(loss=['mean_squared_error'],
              metrics = ['mean_squared_error','mean_absolute_percentage_error'],
              optimizer='adam')
model.fit(X_train, y_train,  
          batch_size=batch_size,      
          epochs=epochs,          
          verbose=1,          
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)

y_pred = model.predict(X_test)
gof = 1 - np.linalg.norm(y_pred - y_test)/np.linalg.norm(y_test - np.mean(y_test))
r_score = r2_score(y_test, y_pred)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('GOF: ', gof)
print('R: ', r_score)


#%%

plt.figure()
plt.plot(y_test, 'k',label = 'True')
plt.plot(y_pred,'b', label = 'Prediction')
plt.legend()
plt.show()
