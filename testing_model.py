# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:45:21 2020

@author: Haseeb Khan
"""
import cv2
import tensorflow as tf
import preparing_data 
import os
from sklearn.metrics import accuracy_score
import numpy as np

CATEGORIES = preparing_data.CATEGORIES
IMG_SIZE = 448
DATADIR = "data/labeled_val"
X_test = []; y_test = []
 
def test_all(path):
    for img in os.listdir(path):
        state = img[-9:-4]
        X_test.append(prepare(path, img))
        y_test.append(label2state_data.state_label(state))
                
def prepare(path, img):
    img_array = cv2.imread(os.path.join(path,img), cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE, IMG_SIZE, 3)

test_all(DATADIR)
model = tf.keras.models.load_model("Imsize448-2-conv-64-nodes-1-dense.model")
y_predict = []

for i in X_test:
    prediction_array = model.predict(i)
    y_predict.append([np.argmax(pred) for pred in prediction_array][0])

print(accuracy_score(y_test, y_predict))