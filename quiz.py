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
    st.title('Computer Science Job Predictor')
    st.write('This app recommends a CS job based on your computer skills and personality.')
    st.write('Rate how true these answers apply on a scale of 1-5 - 1 being strongly disagree and 5 being strongly agree')
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
    questions = {
        "Openness": [
            "I have a vivid imagination",
            "I enjoy trying new things"
        ],
        "Conscientiousness": [
            "I pay attention to details",
            "I get chores done right away"
        ],
        "Extraversion": [
            "I feel comfortable around people",
            "I am the life of the party"
        ],
        "Agreeableness": [
            "I sympathize with others' feelings",
            "I make people feel at ease"
        ],
        "Emotional Range": [
            "I get stressed out easily",
            "I worry about things"
        ],
        "Conversation": [
            "I enjoy discussing ideas",
            "I prefer meaningful conversations"
        ],
        "Openness to Change": [
            "I prefer variety over routine",
            "I adapt easily to change"
        ],
        "Hedonism": [
            "I seek pleasurable activities",
            "I prioritize my own enjoyment"
        ],
        "Self-enhancement": [
            "I strive for personal success",
            "I enjoy being in charge"
        ],
        "Self-transcendence": [
            "I care about all humanity",
            "I help others selflessly"
        ],
        "Role": [
            "I enjoy leadership roles",
            "I adapt my role as needed"
        ]
    }
    
    responses = {}
    for trait in questions:
        st.markdown(f"**{trait}**")
        cols = st.columns(2)
        for i, question in enumerate(questions[trait]):
            with cols[i]:
                key = f"{trait}_{i}"
                responses[key] = st.select_slider(
                    question,
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    key=key
                )
        st.write("---")
    
    if st.button("Submit Personality Assessment"):
        # Calculate scores
        trait_scores = {}
        for trait in questions:
            trait_responses = [v for k, v in responses.items() if k.startswith(trait)]
            trait_scores[trait] = np.mean(trait_responses)
        
        # Display results
        st.success("Assessment complete!")
        st.subheader("Your Personality Scores")
        
        # Display in 3 columns
        cols = st.columns(3)
        for i, (trait, score) in enumerate(trait_scores.items()):
            with cols[i % 3]:
                st.metric(label=trait, value=f"{score:.1f}/5.0")
        
        return trait_scores