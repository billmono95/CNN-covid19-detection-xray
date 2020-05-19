# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 07:56:57 2020

@author: Bill
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random




DATADIR = '/Users/Bill/Desktop/ML AND NN/NUOVOML/Chest_xray_image_dataset_covid_19_and_others'

CATEGORIES = ['COVID19','PNEUMONIA']
data =[] 

def funtion_dir():

  for category in CATEGORIES:
    
    class_index = CATEGORIES.index(category)           
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        new_path = os.path.join(path,img)
        try:
            image_data_temp = cv2.imread(new_path,cv2.IMREAD_GRAYSCALE)
            image_resize = cv2.resize(image_data_temp,(80,80))
        #print(class_index)
        #plt.imshow(image_data_temp)
        #plt.show()
            data.append([image_resize,class_index])
            #print (data)
        except:
            pass
        

funtion_dir()

random.shuffle(data)
print(data[0])    
data = np.asanyarray(data)
print(data[0])

 # Iterate over the Data
x_data =[]
y_data =[]

for x in data:
 x_data.append(x[0])        # Get the X_Data
 y_data.append(x[1])        # get the label


X_Data = np.asarray(x_data) / (255.0)      # Normalize Data
Y_Data = np.asarray(y_data)

X_Data = X_Data.reshape(-1, 80, 80, 1)
print(Y_Data.shape)

print("Save Data.... ")

pickle_out = open('X_Data','wb')
pickle.dump(X_Data, pickle_out)
pickle_out.close()

pickle_out = open('Y_Data','wb')
pickle.dump(Y_Data, pickle_out)
pickle_out.close()

print("Pickled Image Successfully ")