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

from keras.backend.tensorflow_backend import set_session
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

k = 100
#PCA
pca = PCA(n_components = k)
X_new = pca.fit_transform(data_X_1)
ev = pca.explained_variance_ratio_
data_X = np.reshape(X_new,(1747,54,k))

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
batch_size = 32
epochs = 150
#n_dense_1 = 30
#n_dense_2 = 20
#n_dense_3 = 10


model = Sequential()
model.add(Conv1D(filter_depth,
                 kernel_size= a,
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
                 input_shape=(k,54)))
#                 input_shape=(54,k)))

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

initial_weights = model.get_weights()
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

n_splits = 10
frac = 1.0/n_splits*(n_splits - 1)
#kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 7)
#
##kfold = KFold(n_splits = n_splits, shuffle = True, random_state = seed)
##K.get_session().run(tf.global_variables_initializer())
##results = cross_val_score(model, data_X,data_Y, cv = kfold, n_jobs = 4)
##print(results.mean())
#score_vec = np.zeros((n_splits,1))
#it = 0
#for train_idx, test_idx in kfold.split(data_X):
#    K.get_session().run(tf.global_variables_initializer())
#    X_train = data_X[train_idx,:,:]
#    y_train = data_Y[train_idx]
#    X_test = data_X[test_idx,:,:]
#    y_test = data_Y[test_idx]
#    model.fit(X_train, y_train,
#              batch_size=batch_size,
#              epochs=epochs,
#              verbose=1,
#              validation_data=(X_test, y_test))
#    score = model.evaluate(X_test, y_test, verbose=0)
#    score_vec[it] = score[-1]
#    print('R: ', score[-1])
#    it+=1
#

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
#    model.fit(X_train, y_train,
#              batch_size=batch_size,
#              epochs=epochs,
#              verbose=1,
#              validation_data=(X_test, y_test))
#    score = model.evaluate(X_test, y_test, verbose=0)
#    score_vec[i] = score[-1]
#    print('R: ', score[-1])
#
#print(score_vec.mean())


idx = np.random.permutation(N)
idx_ar = np.array(np.array_split(idx,n_splits))
l = np.arange(0,n_splits)
msetot = []
r2tot = []
maetot = []
mapetot = []

for i in range(n_splits):
    train_idx = idx_ar[l[(l < i) | (l > i)]]
    train_idx = np.hstack(train_idx)
    test_idx = idx_ar[l[i]]
    test_idx = np.hstack(test_idx)
    X_train = data_X[train_idx]
    y_train = data_Y[train_idx]
    meany = np.mean(y_train)
    y_train -= meany
    X_test = data_X[test_idx]
    y_test = data_Y[test_idx]
    y_test -= meany
    model.set_weights(initial_weights)
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test))
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test))
    model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.reshape(model.predict(X_test), (len(y_test),1))
    y_test = np.reshape(y_test, (len(y_test),1))
    #plt.figure()
#    plt.scatter(y_pred, y_test)
#    plt.hold(True)
#    plt.plot([-4, 8], [-4,8])
#    plt.axis([-4, 8, -4, 8])
#    idxes = np.abs(y_pred-y_test) > 1.8
#    plt.plot(y_pred[idxes],y_test[idxes], 'ro')
    score_vec[i] = score[-2]
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
