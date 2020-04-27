# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:38:23 2020

@author: Haseeb Khan
"""
import cv2
import numpy as np
import tensorflow as tf
import yaml 
import os
from collections import deque

             
IMG_SIZE = 448
WINDOW_SIZE = 5
path = "data/videos/test"
files = os.listdir(path)
for file in files:
    model = tf.keras.models.load_model("Imsize448-3-conv-64-nodes-1-dense")
    video_ts = file.split('.')[0][-13:]
    out_video_path = 'data/videos/imsize448_label/' + str(video_ts) + '_label.mp4'
    
    CATEGORIES = ["Class1", "Class2", "Class3"]
    cell_file = 'config.yaml'
    with open(cell_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cells = config['cells']
    window = [deque([]) for i in range(len(cells))]
    
    print(out_video_path)
    
    cap = cv2.VideoCapture(os.path.join(path, file))
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(
        *'MP4V'), 1, (1920, 1920)) #Check video resolution 
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        
        for idx, cell in enumerate(cells):
            img_crop = frame[cell['y_min']:cell['y_max'],
                    cell['x_min']:cell['x_max']]
            
            img_crop = cv2.resize(img_crop,(IMG_SIZE, IMG_SIZE))
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            img = img_crop.reshape(-1,IMG_SIZE, IMG_SIZE, 3)
          
            prediction_array = model.predict(img)
            window[idx].append([np.argmax(pred) for pred in prediction_array][0])
            
            if len(window[idx]) > WINDOW_SIZE:
                window[idx].popleft()
            result = max(set(window[idx]), key = window[idx].count)
        
            # Output results to the frame
            frame = cv2.rectangle(
                frame,
                (cell['x_min'], cell['y_min']),
                (cell['x_max'], cell['y_max']),
                (255, 0, 0),
                2)
            cv2.putText(
                frame,
                str(cell['index']),
                (cell['x_min'], cell['y_min'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2)
            cv2.putText(
                frame,
                str(CATEGORIES[result]),
                (cell['x_min'], cell['y_min'] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 0, 255),
                3)
            
        out.write(frame)
    
    cap.release()
    out.release()