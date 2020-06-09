#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:14:00 2020

@author: andrewlin
"""
import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.layers import Conv2D,MaxPooling2D,Flatten, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


# Load the data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the arrays into 4-dimensions to use with Keras
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Normalize the RGB codes
X_train = X_train.astype('float32')
X_train = X_train / 255
X_test = X_test.astype('float32')
X_test = X_test / 255

# Convert the labels to dummy variables via one-hot encoding
y_train= to_categorical(y_train)
y_test = to_categorical(y_test)

# Define an EarlyStopping object to aid in preventing overfitting
stop = EarlyStopping(patience=3)

# A function that creates the model
def model():

    model = Sequential()

    model.add(Conv2D(128, kernel_size=(5,5), input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3,3), input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten()) 
    
    model.add(Dense(128, activation = 'relu'))

    model.add(Dropout(0.21))

    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = model()


model.fit(X_train, y_train, epochs=50, batch_size = 128, validation_split=0.2, callbacks=[stop])
model.evaluate(X_test, y_test)


# Optimizations
#params = {'batch_size': [32, 128, 256], 
          #'learning_rate': [0.1, 0.01, 0.001]}

#random_search = RandomizedSearchCV(model, param_distributions = params, cv = KFold(3))

#random_search.fit(X_train, y_train)








