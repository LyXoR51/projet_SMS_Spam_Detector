# GCP CREDENTIAL for HF SPACE
import os
if "GCP_KEY" in os.environ:
    with open("/tmp/gcp_key.json", "w") as f:
        f.write(os.environ["GCP_KEY"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp_key.json"


#####   LIBRARY  #####
from sqlalchemy import create_engine, text
import streamlit as st
from datetime import datetime
import torch
import tiktoken
import pandas as pd
import requests
import json
import mlflow
import numpy as np
import pandas as pd


st.set_page_config(layout="wide")

#####   DATA & VARIABLE  #####
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE")
ENGINE = create_engine(POSTGRES_DATABASE, echo=True)
TABLE_NAME = 'messages_spam_detector'
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


#####  MLFLOW SERVER  #######
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_name = "SMS_Spam_Detector_NN"
model_version = "latest"
model = mlflow.pytorch.load_model(
    model_uri=f"models:/{model_name}/{model_version}",
    map_location="cpu"
)

##### FUNCTIONS #####

def preprocess(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token = tokenizer.encode(text)
    seq = [token[:25] + [0] * (25 - len(token))]
    return seq

def request_prediction(model, text):
    with torch.no_grad():
        input_tensor = torch.tensor(preprocess(text), dtype=torch.long)
        output = model(input_tensor)
    return 'Spam' if output[0][0] > 0.5 else 'Not Spam'

def update_db(user_input, predict_label, label):

    data = pd.DataFrame([{
        "content": user_input,
        "label_predict": predict_label,
        "label": label,
        "created_at": int(datetime.timestamp(datetime.now()) * 1000)
    }])
    try:
        data.to_sql(name=TABLE_NAME, con=ENGINE, index=False, if_exists="append")
    except Exception as e:
        print("DB update error:", e)

@st.dialog("Model prediction")
def ask_feedback(user_input, predict_label):

    st.markdown(f"# {predict_label} !")
    st.markdown("Help us improve — is your message spam or not?")
    left, right = st.columns(2)
    if left.button("Spam"):
        update_db(user_input, predict_label, "Spam")
        st.rerun()
    if right.button("Not Spam"):
        update_db(user_input, predict_label, "Not Spam")
        st.rerun()

##### APP #####
st.markdown("""
# 📱 Spam Fraud Detector

Welcome to the **Spam Fraud Detector** demo!  
This app automatically classifies SMS messages as *Spam* or *Not Spam* using a trained machine learning model.

Feel free to explore the app — here's what you'll find:

- **🧠 Detector**: Try the model yourself by entering a message and see if it's predicted as spam or not.  
- **📊 Predictions**: Browse past messages and predictions, along with the original training dataset.  
- **🏗️ ML Workflow**: See how the model is built, trained, and improved over time.

# 
            
""")

with st.expander("**Try the Spam Detector**", expanded=True):
    st.markdown("""
    
    Enter a message below to see if our model predicts it as **Spam** or **Not Spam**.  
    After the prediction, please confirm whether the result was correct — your feedback helps the model get smarter over time!
    """)
    user_input = st.text_area('',placeholder='Type here')
    if st.button("Submit"):
        predict_label = request_prediction(model, user_input)
        ask_feedback(user_input, predict_label)

        
with st.expander("**Predictions & Data**"):
    st.markdown("""

Here you can explore the predictions made by the model, along with the corresponding message content.  
This section also includes the original training dataset, allowing you to compare real data with recent user inputs and predictions.
""")
    tab1, tab2 = st.tabs([" 🤖 Predictions", " 🗄️ Global"])
    with tab1:
        with ENGINE.connect() as conn:
            stmt = text(f"""
                SELECT * 
                FROM {TABLE_NAME}
                WHERE label_predict IS NOT NULL
                ORDER BY id DESC 
    """)       
            result = conn.execute(stmt)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            df['created_at'] = pd.to_datetime(df['created_at'], unit='ms')
            st.dataframe(df)


    with tab2:
        with ENGINE.connect() as conn:
            stmt = text(f"""
                SELECT * 
                FROM {TABLE_NAME}
                ORDER BY id DESC  
    """)
            result = conn.execute(stmt)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            df['created_at'] = pd.to_datetime(df['created_at'], unit='ms')
            st.dataframe(df)


with st.expander("**ML Workflow**"):
    st.markdown("""    
                
    This project follows a complete MLOps lifecycle — from data preparation to model deployment and continuous improvement.""")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, "images", "mlworkflow.png")

    st.image(image_path)
    st.markdown("""

    ####  1. Data Preparation (ETL)
    - The original SMS dataset (CSV) was cleaned and transformed through an ETL pipeline.  
    - Processed data was stored in a **SQL database** for easy access and tracking.

    ####  2. Model Training
    - Several training experiments were run on a **Kubernetes cluster** using **KubeRay**, **MLflow**, and **PyTorch**.  
    - **MLflow Tracking** was used to log experiments, metrics, and model versions.  
    - The best-performing model was registered in the **MLflow Model Registry**.

    ####  3. Deployment
    - The final model was deployed via an **MLflow Model Server** hosted on **Hugging Face Spaces**.  
    - A **Streamlit app** provides the user interface for real-time predictions.

    ####  4. Continuous Feedback Loop
    - Users can test the model by submitting new messages.  
    - After each prediction (*Spam / Not Spam*), they can confirm or correct the result.  
    - These validated samples are stored in the database for **future retraining and model improvement**.

    ####  5. Next Steps : Full MLOps
    - **Apache Airflow :** to orchestrate data pipelines and automate retraining.  
    - **Evidently AI :** to monitor data drift and model performance over time.  
    - **Jenkins :** to implement CI/CD for code, model, and deployment automation.  

    These future integrations will enable a fully **automated, monitored, and self-improving ML system**.
    """)