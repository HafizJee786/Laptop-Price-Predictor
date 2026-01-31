import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Load the Model and Data
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    df = pickle.load(open('df.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'pipe.pkl' or 'df.pkl' file not found. Please export these files from your notebook first.")

# Page Configuration
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")
st.title("💻 Laptop Price Predictor")

# 2. UI Layout (Two Columns for Input Fields)
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox('Laptop Brand', df['brand'].unique())
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    ram_type = st.selectbox('RAM Type', df['Ram_type'].unique())
    cpu = st.selectbox('CPU Brand', df['cpu_brand'].unique())
    gpu = st.selectbox('GPU Brand (Graphics Card)', df['GPU'].unique())

with col2:
    spec_rating = st.number_input('Spec Rating (0-100)', min_value=0, max_value=100, value=70)
    rom = st.selectbox('Storage (in GB)', [128, 256, 512, 1024, 2048])
    rom_type = st.selectbox('Storage Type', df['ROM_type'].unique())
    os = st.selectbox('Operating System', df['OS'].unique())
    warranty = st.selectbox('Warranty (Years)', [0, 1, 2, 3])

ppi = st.number_input('PPI (Pixels Per Inch)', value=141.2)

st.markdown("---")

# 3. Prediction Logic
if st.button('Predict Laptop Price'):
    # Create input dictionary (Ensure column names match X_train)
    input_data = {
        'brand': [brand],
        'spec_rating': [spec_rating],
        'Ram': [ram],
        'Ram_type': [ram_type],
        'ROM': [rom],
        'ROM_type': [rom_type],
        'GPU': [gpu],
        'OS': [os],
        'warranty': [warranty],
        'ppi': [ppi],
        'cpu_brand': [cpu]
    }
    
    # Convert dictionary to DataFrame
    query_df = pd.DataFrame(input_data)
    
    # Perform Prediction 
    # (Note: Using np.exp because the model was likely trained on log-transformed prices)
    try:
        prediction = np.exp(pipe.predict(query_df))
        st.header(f"💰 Estimated Price: Rs. {int(prediction[0]):,}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
