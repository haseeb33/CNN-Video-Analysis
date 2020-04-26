# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:33:55 2020

@author: Haseeb Khan
"""
from __future__ import absolute_import, division, print_function
from tqdm import tqdm
from numpy.random import randn
import pathlib
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from matplotlib.image import imread

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
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
     
    return image,y
  
def _input_fn(x,y):
    ds=tf.data.Dataset.from_tensor_slices((x,y))
    ds=ds.map(_parse_data)
    ds=ds.shuffle(buffer_size=data_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
  
    return ds
  
train_ds=_input_fn(x_train,y_train)
validation_ds=_input_fn(x_test,y_test)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
VGG16_MODEL=tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
VGG16_MODEL.trainable=False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = tf.keras.layers.Dense(len(label_names),activation='softmax')

model = tf.keras.Sequential([VGG16_MODEL,global_average_layer, prediction_layer])

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy',metrics=["accuracy"])
# Increase epochs maybe around 100 and check again
history = model.fit(train_ds,epochs=100, validation_steps=2, steps_per_epoch =2, validation_data=validation_ds)
model.save("Imsize448_VGG16_CNN.model")
validation_steps = 1
loss0,accuracy0 = model.evaluate(validation_ds, steps = validation_steps)
 
print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

