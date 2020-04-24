# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:17:29 2020

@author: Haseeb Khan
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

DATADIR = "data/labeled_train"
IMG_SIZE = 448
CATEGORIES = ["Class1", "Class2", "CLass3"]

training_data = []
def create_training_data():
    for category in CATEGORIES:  
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except:
                None
            
create_training_data()
random.shuffle(training_data)
X=[]; y=[]
for features, label in training_data:
    X.append(features); y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
y = np.array(y)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()