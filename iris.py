# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:31:52 2021

@author: Suppi's_space
"""

import pandas as pd
import seaborn as sns
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')
st.title("Flower Prediction")
st.write("Here in this project we will deploy iris flower data set")
df=sns.load_dataset('iris')
if st.checkbox("Show Dataset"):
    df

with open('E:/GC/knn_transform','rb') as file:
    tv=pickle.load(file)
with open('E:/GC/knnbrain','rb') as file:
    pred=pickle.load(file)

    
a=st.text_input("Sepal Length",0)
b=st.text_input("Sepal Width",0)
c=st.text_input("Petal Length",0)
d=st.text_input("Petal Width",0)

x=tv.transform([[a,b,c,d]])
result=pred.predict([[x[0][0],x[0][1],x[0][2],x[0][3]]])
st.write(result[0])
