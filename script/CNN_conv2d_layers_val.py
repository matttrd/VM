"""
Application 2D convulution where input data is seen as an "image", i.e. a function f:R^2 -> R
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.signal import detrend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, AveragePooling1D, BatchNormalization
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers import Flatten
from keras.utils import np_utils
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
import keras.backend as K
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split as split
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.35
set_session(tf.Session(config=config))

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

#idx = data_Y > 60
#data_Y = data_Y[idx]
#y_offset = 70
#data_Y = data_Y - y_offset
#data = data_X[:-sum(np.logical_not(idx))[0]]

idx = data_Y > 60
data_Y = data_Y[idx]
data_Y = data_Y - 70
idx = np.reshape(idx, (1771))
data_X = data_X[idx]

#%%
data_X_1 = np.reshape(data_X,(1747*54,2048))
#k = 100
#f = SelectKBest(f_regression, k=k)
#X_new = f.fit_transform(data_X_1, np.repeat(data_Y,54))

k = 50
#PCA
pca = PCA(n_components = k)
X_new = pca.fit_transform(data_X_1)
ev = pca.explained_variance_ratio_
data_X = np.reshape(X_new,(1747,54,k,1))

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
data_X = data_X.swapaxes(2,1)
#X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.25, random_state=seed)

# %% CNN Model
# CNN model for regression
filter_depth = 4
filter_depth2 = 8
filter_depth3 = 16
a = 2
a2 = 2
a3 = 2
am = 2
am2 = 2
am3 = 2
#n_dense_1 = 30
#n_dense_2 = 20
#n_dense_3 = 10


model = Sequential()
model.add(Conv2D(filter_depth,
                 kernel_size= (4,16),
                 activation='relu',
                 strides = 2,
                 padding='same',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
#                 kernel_regularizer=None,
#                 bias_regularizer=None,
#                 activity_regularizer=None,
#                 kernel_constraint=None,
#                 bias_constraint=None,
                 input_shape=(k,54, 1)))
#                 input_shape=(54,k)))
#model.add(BatchNormalization(axis = -1))
model.add(AveragePooling2D(pool_size=(am,am)))
model.add(Conv2D(filter_depth2,
                 kernel_size=(4,8),
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
model.add(AveragePooling2D(pool_size=(am2,am2)))
model.add(Conv2D(filter_depth3,
                 kernel_size=(3,3),
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
#model.add(AveragePooling1D(pool_size=am3))
model.add(Flatten())
model.add(Dropout(0.1))
#model.add(Dense(n_dense_1, activation='relu'))
#model.add(Dropout(0.1))
#model.add(Dropout(0.1))
#model.add(Dense(n_dense_2, activation='relu'))
model.add(Dense(1, kernel_initializer='normal',
                use_bias = True,
                activation = None))

layer = model.layers[3].output
layer = Flatten()(layer)
layer = Dropout(0.1)(layer)
layer = Dense(1, kernel_initializer='normal',
                use_bias = True,
                activation = None)(layer)
model_2 = Model(model.layers[0].input, layer)

layer = model.layers[1].output
layer = Flatten()(layer)
layer = Dropout(0.1)(layer)
layer = Dense(1, kernel_initializer='normal',
                use_bias = True,
                activation = None)(layer)
model_1 = Model(model.layers[0].input, layer)

model.summary()
model_1.summary()
model_2.summary()

def r_score_metric(y_true, y_pred):
    y_true = K.tensorflow_backend._to_tensor(y_true, dtype = tf.float32)
    y_pred = K.tensorflow_backend._to_tensor(y_pred, dtype = tf.float32)
#    return  1 - K.sum(y_true - y_pred)
    return  1 - K.sum((y_true - y_pred)**2)/(K.sum((y_true - K.mean(y_true))**2))

def max_error(y_true, y_pred):
    tnf = K.tensorflow_backend
    y_true = tnf._to_tensor(y_true, dtype = tf.float32)
    y_pred = tnf._to_tensor(y_pred, dtype = tf.float32)
    return K.max(K.abs((y_true-y_pred)))

def mape(y_true, y_pred):
    tnf = K.tensorflow_backend
    y_true = tnf._to_tensor(y_true, dtype = tf.float32)
    y_pred = tnf._to_tensor(y_pred, dtype = tf.float32)
    return K.mean(K.abs((y_true-y_pred)/y_true))

model.compile(loss=['mean_squared_error'],
              metrics = ['mean_squared_error', r_score_metric, max_error, mape],
              optimizer='adam')

model_1.compile(loss=['mean_squared_error'],
              metrics = ['mean_squared_error', r_score_metric, max_error, mape],
              optimizer='adam')
model_2.compile(loss=['mean_squared_error'],
              metrics = ['mean_squared_error', r_score_metric, max_error, mape],
              optimizer='adam')

initial_weights = model.get_weights()
initial_weights_1 = model_1.get_weights()
initial_weights_2 = model_2.get_weights()
models = [model, model_1, model_2]
initial_weights_vec = [initial_weights, initial_weights_1, initial_weights_2]
#%%
# Compile model

#
#model = KerasRegressor(build_fn = create_model(filter_depth, filter_depth2, filter_depth3,a,a2,a3,am,
#                 am2,am3,n_dense_1, k = k),
#                        epochs = epochs,
#                        batch_size = batch_size,
#                        verbose = 1)
#

#seed = 1337
#np.random.seed(seed)

K1 = 100
K2 = 10
q1 = 0.3
msetot = []
r2tot = []
maetot = []
mapetot = []
msetot_val = np.zeros((K2, 3))
model_idx = []
batch_size = 32
epochs = 400

for k1 in range(K1):
    print("Cross validation: " + str(k1) + "/" + str(K1))
    X_train, X_test, y_train, y_test = split(data_X, data_Y, test_size = q1)
    meany = np.mean(y_train)
    y_train -= meany
    y_test -= meany
    for k2 in range(K2):
        print("Inner validation: " + str(k2) + "/" + str(K2))
        X_train_val, X_val, y_train_val, y_val = split(X_train, y_train, test_size = q1)
        for m in range(3):
            mod = models[m]
            mod.set_weights(initial_weights_vec[m])
            mod.fit(X_train_val, y_train_val,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=0,
                      validation_data=(X_val, y_val))
            mod.fit(X_train_val, y_train_val,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=0,
                      validation_data=(X_val, y_val))
            score = mod.evaluate(X_val, y_val, verbose=0)
            msetot_val[k2,m] = score[0]
    meanmse = np.mean(msetot_val, axis=0)
    min_index = np.argmin(meanmse)
    model_idx.append(min_index)
    print(meanmse)
    #plt.figure()
#    plt.scatter(y_pred, y_test)
#    plt.hold(True)
#    plt.plot([-4, 8], [-4,8])
#    plt.axis([-4, 8, -4, 8])
#    idxes = np.abs(y_pred-y_test) > 1.8
#    plt.plot(y_pred[idxes],y_test[idxes], 'ro')
    mod = models[min_index]
    mod.fit(X_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=0,
                      validation_data=(X_test, y_test))
    mod.fit(X_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=0,
                      validation_data=(X_test, y_test))
    score = mod.evaluate(X_test, y_test, verbose=0)
    msetot.append(score[0])
    r2tot.append(score[2])
    maetot.append(score[3])
    mapetot.append(score[4])

    print('R: ', score[2])
    print('MSE:', score[0])
    print('max_error:', score[3])
    print('mape:', score[4])
#%%

plt.figure()
y_pred = model.predict(X_test)
plt.scatter(y_pred,y_test)

print(str(np.array(msetot).mean()) + " " +
" " + str(np.array(msetot).std()) +
" " + str(np.array(r2tot).mean()) +
" " + str(np.array(r2tot).std()) +
" " + str(np.array(maetot).mean()) +
" "  + str(np.array(maetot).std()) +
" " + str(np.array(mapetot).mean()) +
" " + str(np.array(mapetot).std()))

np.savetxt('mse.csv', msetot, delimiter=',')
np.savetxt('r2.csv', r2tot, delimiter=',')
np.savetxt('mae.csv', maetot, delimiter=',')
np.savetxt('mape.csv', mapetot, delimiter=',')
np.savetxt('mseval.csv', msetot_val, delimiter=',')
