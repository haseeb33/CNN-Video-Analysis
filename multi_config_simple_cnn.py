# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:06:14 2020

@author:  Haseeb Khan
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle

X = pickle.load(open("pickle_files/X.pickle","rb"))
y = pickle.load(open("pickle_files/y.pickle","rb"))
X = X/255.0
CATEGORIES = 3

dense_layers = [1,2,3]
layer_sizes = [32,64,128,256]
conv_layers = [3,4,5]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "Imsize448-{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)
            
            model = Sequential()
            
            model.add(Conv2D(layer_size, kernel_size=(5, 5), activation="relu", input_shape=X.shape[1:]))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size,  kernel_size=(3, 3), activation="relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))
            
            model.add(Flatten()) 
            
            for _ in range(dense_layer):
                model.add(Dense(layer_size, activation='relu'))
                model.add(Dropout(0.5))
            
            model.add(Dense(CATEGORIES, activation='softmax'))
            
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
            
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            
            model.fit(X, y, batch_size=16, epochs=10, validation_split=0.2, callbacks=[tensorboard])
            model.save("models/{}.model".format(NAME))