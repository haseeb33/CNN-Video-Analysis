# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:09:21 2020

@author: Haseeb Khan
"""
from __future__ import absolute_import, division, print_function

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pathlib
import random
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
data_dir = pathlib.Path("data/labeled_train")

label_names={'Class1': 0, 'Class2': 1, 'Class3': 2}
label_key=['Class1','Class2','Class3']
all_images = list(data_dir.glob('*/*'))
all_images = [str(path) for path in all_images]
random.shuffle(all_images)
all_labels=[label_names[pathlib.Path(path).parent.name] for path in all_images]
data_size=len(all_images)
train_test_split=(int)(data_size*0.2)

x_train=all_images[train_test_split:]
x_test=all_images[:train_test_split]
y_train=all_labels[train_test_split:]
y_test=all_labels[:train_test_split]

IMG_SIZE=448 
BATCH_SIZE = 16

def _parse_data(x,y):
    image = tf.io.read_file(x)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
     
    return image,y

def augment(image,label):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3) # Random brightness

    return image,label

def _input_fn(x,y, val=0):
    ts = tf.data.Dataset.from_tensor_slices((x,y))
    if val==0:
        ds1 = ts.map(_parse_data)
        ds2 = ts.map(augment)
        ds = ds1.concatenate(ds2)
    else:
        ds = ts.map(_parse_data)
    ds=ds.shuffle(buffer_size=data_size*2)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

train_ds=_input_fn(x_train,y_train)
validation_ds=_input_fn(x_test,y_test, val=1)
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
dense_layers = [1,2,3] #,2,3
layer_sizes = [32,64,128,256] #32,64,128,256
conv_layers = [3,4,5] #3,4

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = f"Augment-Imsize{IMG_SIZE}-{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense"
            print(NAME)
            
            model = Sequential()
            
            model.add(Conv2D(layer_size, kernel_size=(5, 5), activation="relu", input_shape=IMG_SHAPE))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size,  kernel_size=(3, 3), activation="relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))
            
            model.add(Flatten()) 
            
            for _ in range(dense_layer):
                model.add(Dense(layer_size, activation='relu'))
                model.add(Dropout(0.25))
            
            model.add(Dense(3, activation='softmax'))
            
            tensorboard = TensorBoard(log_dir="CNN_models/logs/{}".format(NAME))
            
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            
            # steps_per_epoch value should be total image size/batch size
            model.fit(train_ds, epochs=10, validation_steps=2, steps_per_epoch =512, validation_data=validation_ds, callbacks=[tensorboard])
            print('Summary',model.summary())
            model.save("CNN_models/models/{}".format(NAME))
