batch_size = 32
epochs = 30
n_dense_1 = 10
n_dense_2 = 50
#n_dense_3 = 10
kernel_size = int(1)
filter_depth = int(16)
pool_size = int(2)
#stride = int(kernel_size/2)
stride = 1
model = Sequential()
model.add(Conv1D(filter_depth,
                 kernel_size= kernel_size,
                 activation='relu', 
                 strides=stride, 
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
model.add(AveragePooling1D(pool_size = pool_size))

model.add(Conv1D(int(filter_depth/2),
#                 kernel_size= int(kernel_size/2),
                 kernel_size= kernel_size,
                 activation='relu', 
#                 strides=int(stride/2), 
                 strides = stride,
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
model.add(AveragePooling1D(pool_size=pool_size))

model.add(Conv1D(int(filter_depth/4),
#                 kernel_size= int(kernel_size/4),
                 kernel_size= kernel_size,
                 activation='relu', 
#                 strides=int(stride/4), 
                 strides = stride,                
                 padding='same', 
                 use_bias=True, 
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros', 
#                 kernel_regularizer=None, 
#                 bias_regularizer=None, 
#                 activity_regularizer=None, 
#                 kernel_constraint=None, 
#                 bias_constraint=None,
))
#model.add(BatchNormalization(axis = -1))
model.add(AveragePooling1D(pool_size=pool_size))

#model.add(Conv1D(int(filter_depth/8),
##                 kernel_size= kernel_size/8,
#                 kernel_size= kernel_size,
#                 activation='relu', 
##                 strides=stride/8, 
#                 strides = stride,    
#                 padding='same', 
#                 use_bias=True, 
#                 kernel_initializer='glorot_uniform', 
#                 bias_initializer='zeros', 
##                 kernel_regularizer=None, 
##                 bias_regularizer=None, 
##                 activity_regularizer=None, 
##                 kernel_constraint=None, 
##                 bias_constraint=None,
#))
##model.add(BatchNormalization(axis = -1))
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