# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 21:56:09 2022

@author: Superuser
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('C:/Users/Superuser/Desktop/Fatima/ML Projects/Breast Cancer Classification/trained_model.sav', 'rb'))

input_data = (13.08,15.71,85.63,520,0.1075,0.127,0.04568,0.0311,0.1967,0.06811,0.1852,0.7477,1.383,14.67,0.004097,0.01898,0.01698,0.00649,0.01678,0.002425,14.5,20.49,96.09,630.5,0.1312,0.2776,0.189,0.07283,0.3184,0.08183)

# changing the input data into a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The Breast Cancer is Malignant')
else:
    print('The Breast Cancer is Benign')