# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:33:19 2020

@author: Bill
"""
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten,Conv2D,MaxPooling2D

import pickle

X_Data = open('X_Data','rb')
X = pickle.load(X_Data)

Y_Data = open('Y_Data','rb')
y = pickle.load(Y_Data)


X_prova = open('X_prova','rb')
Xp = pickle.load(X_prova)

Y_prova = open('Y_prova','rb')
yp = pickle.load(Y_prova)
print('Reading Dataset from Pickle Object')

#print(X.shape,y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#print(y_test)
#IMPORTANTE SERVE PER PREDIRE UNA SINGOLA IMMAGINE

#X_tes = np.expand_dims(X_test[2], 0)
#print(model.predict(X_tes))
#print(X_tes.shape)
#print(y_test[2])



model = Sequential()
model.add(Conv2D(150, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(75, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, y_train,epochs=5)

W = model.evaluate(X_test,y_test)
#if(W[1]>= 0.94):
#    model.save("model.h5")
print(W)
print(model.summary())

'''
T = model.evaluate(Xp,yp)
print(T)

'''

