#%%writefile streamlit_app/app.py

# Import libraries
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ----- SECTION 2: CREATE THE STREAMLIT APP -----

# Create a directory for the Streamlit app
#!mkdir -p streamlit_app
# use "python -m streamlit run app.py" if streamlit not running through terminal


class SimpleClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)         # Second fully connected layer
        self.fc3 = nn.Linear(64, n_classes)   # Output layer
        self.relu = nn.ReLU()                  # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation function here, as we'll use CrossEntropyLoss
        return x

# Load the saved model
with open('job_model.pkl', 'rb') as f:
    model = pickle.load(f)
#print(model.forward(torch.tensor([6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0.714603305,0.480564217,0.470876568,0.039610646,0.742567413,0.086926736,0.339776151,0.091158037,0.230518077,0.208745599])))

target_names = ['AI ML Specialist', 'API Specialist', 'Application Support Engineer',
 'Business Analyst', 'Customer Service Executive',
 'Cyber Security Specialist', 'Database Administrator', 'Graphics Designer',
 'Hardware Engineer', 'Helpdesk Engineer', 'Information Security Specialist',
 'Networking Engineer', 'Project Manager', 'Software Developer',
 'Software tester', 'Technical Writer']

input_list = ['Database Fundamentals','Computer Architecture','Distributed Computing Systems','Cyber Security',
              'Networking','Software Development','Programming Skills','Project Management','Computer Forensics Fundamentals',
              'Technical Communication','AI ML','Software Engineering','Business Analysis','Communication skills','Data Science',
              'Troubleshooting skills','Graphics Designing','Openness','Conscientousness','Extraversion','Agreeableness',
              'Emotional_Range','Conversation','Openness to Change','Hedonism','Self-enhancement','Self-transcendence','Role']

# Set up the Streamlit page
st.title('Computer Science Job Predictor')
st.write('This app recommends a CS job based on your computer skills and personality.')

# Create input sliders for the features
st.sidebar.header('Computer Science Skills')

def user_input_features():
    thing = []
    data = {}
    for i in range(len(input_list) - 1):
        if i == 17:
            st.sidebar.header('Personality Scores')
        if i < 17:
            thing.append(st.sidebar.select_slider(input_list[i], options=[0,1,2,3,4,5,6], value=3))
        else:
            thing.append(st.sidebar.slider(input_list[i], 0.0, 1.0, 0.5))
        data[input_list[i]] = thing[i]   

    features = pd.DataFrame(data, index=[0])
    return features

# Display the user input
user_input = user_input_features()
st.subheader('User Input:')
st.write(user_input)

# Make prediction
prediction_proba = model.forward(torch.tensor(user_input.values.astype(np.float32)))
prediction_highval = torch.argmax(prediction_proba)
other_preds_val = torch.topk(prediction_proba,4)
prediction = target_names[prediction_highval]
other_preds = []
for i in other_preds_val[1]:
    other_preds.append(target_names[i])

# Display the prediction
st.subheader('Prediction:')
st.write(f'Your future job: **{prediction}**')
st.write(f'Other recommendations:')
for i in other_preds:
    st.write(f' - **{i}**')

# Display prediction probabilities
st.subheader('Prediction Probability:')
prob_df = pd.DataFrame(prediction_proba.detach().numpy(), columns=target_names)
st.write(prob_df)

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
