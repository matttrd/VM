# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 18:15:52 2017

@author: Matteo
"""
from keras.layers.convolutional import Conv1D
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers import Flatten
from keras.utils import np_utils
from sklearn import preprocessing
import pandas as pd
from random import shuffle

with open('y_noised_class1.pickle', 'rb') as fl:
    y1 = pickle.load(fl)
 
with open('y_noised_class2.pickle', 'rb') as fl:
    y2 = pickle.load(fl)

with open('y_noised_class3.pickle', 'rb') as fl:
    y3 = pickle.load(fl)
   
y = np.concatenate((y1,y2,y3))

classes = np.concatenate((np.ones(len(y1),dtype = np.int8),np.ones(len(y2),\
                                  dtype = np.int8) + 1,np.ones(len(y3),dtype = np.int8) + 2))

l = list(zip(y,classes))
shuffle(l)
y_s, c_s = zip(*l)
y_s = np.array(y_s)
c_s = np.array(c_s)

model_cnn = Sequential()
pool_length = 8
nb_filter = 16

#keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='float16',
#    padding='pre', truncating='pre', value=0.)
model_cnn.add(Conv1D(nb_filter = nb_filter,
                        filter_length = 6,
                        border_mode='same',
                        activation='relu',
                        input_shape = (100,1)))
model_cnn.add(Conv1D(nb_filter = nb_filter,
                        filter_length = 6,
                        border_mode='same',
                        activation='relu'))
model_cnn.add(MaxPooling1D(pool_length=pool_length))
model_cnn.add(Conv1D(nb_filter = nb_filter,
                        filter_length = 12,
                        border_mode='same',
                        activation='relu'))
model_cnn.add(Conv1D(nb_filter = nb_filter,
                        filter_length = 4,
                        border_mode='same',
                        activation='relu'))
model_cnn.add(MaxPooling1D(pool_length=pool_length))
model_cnn.add(Flatten())
model_cnn.add(Dense(3,activation='softmax'))
model_cnn.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_cnn.summary()


# %%
p = 0.5

def class_to_one_hot(y,categories_number):
    cat = np.arange(categories_number)
    return np.array([e==cat for e in y]).astype(int)

X_train = y_s[0:int(p*len(y_s))]
X_test = y_s[int(p*len(y_s)) + 1:]
label_train = c_s[0:int(p*len(y_s))]
label_test = c_s[int(p*len(y_s)) + 1:] 
y_train = class_to_one_hot(label_train,3)
y_test = class_to_one_hot(label_test,3)
X_train2 = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test2 = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

# %%                
batch_size = 100                
print('Train CNN...')
model_cnn.fit(X_train2, y_train, 
          batch_size=batch_size,
          nb_epoch=10,
          validation_data=(X_test2, y_test),shuffle = False)
score_cnn, acc_cnn = model_cnn.evaluate(X_test2, y_test, 
                            batch_size=32)
print('Test score:', score_cnn)
print('Test accuracy:', acc_cnn)
