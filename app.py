#%%writefile streamlit_app/app.py

# ask whether to take a personality quiz or to manually insert all scores
# if personality quiz, go to quiz.py
# if manually insert, go to app.py

# Import libraries
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from streamlit_option_menu import option_menu

class SimpleClassifier(nn.Module):
   def __init__(self, n_features, n_classes):
       super(SimpleClassifier, self).__init__()
       self.fc1 = nn.Linear(n_features, 128)  # First fully connected layer
       self.bn1 = nn.BatchNorm1d(128)
       self.fc2 = nn.Linear(128, 64)         # Second fully connected layer
       self.bn2 = nn.BatchNorm1d(64)
       self.fc3 = nn.Linear(64, n_classes)   # Output layer
       self.relu = nn.ReLU()                  # Activation function
       self.dropout = nn.Dropout(0.3)


   def forward(self, x):
       x = self.bn1(self.relu(self.fc1(x)))
       x = self.dropout(x)
       x = self.bn2(self.relu(self.fc2(x)))
       x = self.fc3(x)  # No activation function here, as we'll use CrossEntropyLoss
       return x
   
import manual



# ----- SECTION 2: CREATE THE STREAMLIT APP -----

# Create a directory for the Streamlit app
#!mkdir -p streamlit_app
# use "python -m streamlit run app.py" if streamlit not running through terminal
# Set up the Streamlit page
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    .stSlider {
        color: #0d6efd;
    }
    .st-b7 {
        color: #212529;
    }
    .st-bb {
        background-color: #ffffff;
    }
    .job-card {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #0d6efd;
    }
    .prediction-header {
        color: #0d6efd;
    }
    .skill-header {
        color: #495057;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Computer Science Job Predictor')
st.write('This app recommends a CS job based on your computer skills and personality.')

st.sidebar.success("Select a page above.")

class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='Menu',
                options=['Manual'],
                icons=['house-fill'],
                menu_icon='chat-text-fill',
                default_index=0,
                )

        if app == "Manual":
            manual.app()
             
    run()            

# Add a sample image based on prediction
#st.subheader('Iris Type:')
#if prediction[0] == 0:
#    st.write('Iris Setosa')
#    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/440px-Kosaciec_szczecinkowaty_Iris_setosa.jpg', width=300)
#elif prediction[0] == 1:
#    st.write('Iris Versicolor')
#    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/440px-Iris_versicolor_3.jpg', width=300)
#else:
#    st.write('Iris Virginica')
#    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/440px-Iris_virginica.jpg', width=300)

# ----- SECTION 3: RUN THE STREAMLIT APP IN COLAB -----
import os

#pip3 install -q streamlit
#!npm install localtunnel
#!wget -q -O - ipv4.icanhazip.com # USE THIS OUTPUT (ex., '34.133.84.111') as the Tunnel Password if you are asked on the next page
#!streamlit run streamlit_app/app.py & npx localtunnel --port 8501
