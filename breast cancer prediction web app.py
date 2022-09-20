# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 22:10:34 2022

@author: Superuser
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/Superuser/Desktop/Fatima/ML Projects/Breast Cancer Classification/trained_model.sav', 'rb'))


# creating a function for prediction

def breast_cancer_prediction(input_data):
    
    # changing the input data into a numpy array
    input_data_as_numpy_array = np.asarray(input_data,dtype=float)
    
    # reshape the numpy array as we are predicting for one datapoint
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
        return 'The Breast Cancer is Malignant'
    else:
        return 'The Breast Cancer is Benign'
    

def main():
    
    # giving a title
    st.title('Breast Cancer Prediction Web App')
    
    # getting the input data from the user
   #  	mean fractal dimension	...	worst radius	worst texture	worst perimeter	worst area	worst smoothness	worst compactness	worst concavity	worst concave points	worst symmetry	worst fractal dimension
    
    
    radius_mean = st.text_input('Mean Radius ')
    texture_mean = st.text_input('Mean Texture ')
    perimeter_mean = st.text_input('Mean Perimeter ')
    area_mean = st.text_input('Mean Area')
    smoothness_mean = st.text_input('Mean Smoothness')
    compactness_mean = st.text_input('Mean compactness')
    concavity_mean = st.text_input('Mean concavity')
    concave_points_mean = st.text_input('Mean concave points')
    symmetry_mean = st.text_input('Mean Symmetry')
    fractal_dimension_mean = st.text_input('Mean Fractional Dimension')
    radius_se = st.text_input('Mean_se Radius')
    texture_se = st.text_input('Mean_se Texture')
    perimeter_se = st.text_input('Mean_se Perimeter')
    area_se = st.text_input('Mean_se Area')
    smoothness_se = st.text_input('Mean_se Smoothness')
    compactness_se = st.text_input('Mean_se Compactness')
    concavity_se = st.text_input('Mean_se Concavity')
    concave_points_se = st.text_input('Mean_se Concave Points')
    symmetry_se = st.text_input('Mean_se Symmetry')
    fractal_dimension_se = st.text_input('Mean_se Fractional Dimension')
    radius_worst = st.text_input('Worst Radius')
    texture_worst = st.text_input('Worst Texture')
    perimeter_worst = st.text_input('Worst Perimeter')
    area_worst = st.text_input('Worst Area')
    smoothness_worst = st.text_input('Worst Smoothness')
    compactness_worst = st.text_input('Worst Compactness')
    concavity_worst = st.text_input('Worst Concavity')
    concave_points_worst = st.text_input('Worst Concave Points')
    symmetry_worst = st.text_input('Worst Symmetry')
    fractal_dimension_worst = st.text_input('Worst Fractional Dimension')
    
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    
    if st.button('Breast Cancer Test Result'):
        diagnosis = breast_cancer_prediction([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst])
    
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
    
    
    