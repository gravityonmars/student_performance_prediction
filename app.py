import streamlit as st
import joblib
import pandas as pd 

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Student Performance Prediction")