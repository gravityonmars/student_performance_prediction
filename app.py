import joblib
import pandas as pd
import streamlit as st

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('Student Performance Prediction')

expected_features = scaler.feature_names_in_

inputs = [
    st.number_input(f'{col}', value=0.0) for col in expected_features
]

if st.button('Predict'):
  input_df = pd.DataFrame([inputs], columns=expected_features)
  scaled_data = scaler.transform(input_df)
  prediction = model.predict(scaled_data)[0]

  st.success(f'Predicted Student Performance: {prediction}')