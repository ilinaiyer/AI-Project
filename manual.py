import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def app():

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


    # Create input sliders for the features
    col1, col2, col3 = st.columns([3,3,2])
    with col1:
        st.header('Computer Science Skills')

        def user_input_features1():
            thing = []
            data = {}

            for i in range(17):
                thing.append(st.select_slider(input_list[i], options=[0,1,2,3,4,5,6], value=3))
                data[input_list[i]] = thing[i]

            features = pd.DataFrame(data, index=[0])
            return features,data
        # Display the user input
        user_input1,data1 = user_input_features1()
        st.subheader('User Input:')
        st.write(user_input1)
    with col2:
        st.header('Personality Scores')

        def user_input_features2():
            thing = []
            data = {}

            for i in range(17,len(input_list)-1):
                thing.append(st.slider(input_list[i], 0.0, 1.0, 0.5))
                data[input_list[i]] = thing[i-17]

            features = pd.DataFrame(data, index=[0])
            return features,data
        # Display the user input
        user_input2,data2 = user_input_features2()
        st.subheader('User Input:')
        st.write(user_input2)
    with col3:
        data = data1 | data2
        user_input = pd.DataFrame(data, index=[0])
        # Make prediction
        #with st.spinner('Analyzing your profile...'):
        prediction_proba = model.forward(torch.tensor(user_input.values.astype(np.float32)))
        prediction_highval = torch.argmax(prediction_proba)
        prediction = target_names[prediction_highval]

        # Display the prediction
        # LINE 128
        st.subheader('Prediction:')
        st.write(f'Your future job: **{prediction}**')

        # Display prediction probabilities
        st.subheader('Prediction Probability:')
        prob_df = pd.DataFrame(prediction_proba.detach().numpy(), columns=target_names)
        st.write(prob_df)