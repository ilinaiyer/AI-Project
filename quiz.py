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
    st.write('Rate how true these answers apply on a scale of 1-5 - (1 being strongly disagree and 5 being strongly agree). These values can be used on the manual page')
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
        "1": [
            "I have a vivid imagination",
            "I enjoy trying new things"
        ],
        "2": [
            "I pay attention to details",
            "I get chores done right away"
        ],
        "3": [
            "I feel comfortable around people",
            "I am the life of the party"
        ],
        "4": [
            "I sympathize with others' feelings",
            "I make people feel at ease"
        ],
        "5": [
            "I get stressed out easily",
            "I worry about things"
        ],
        "6": [
            "I enjoy discussing ideas",
            "I prefer meaningful conversations"
        ],
        "7": [
            "I prefer variety over routine",
            "I adapt easily to change"
        ],
        "8": [
            "I seek pleasurable activities",
            "I prioritize my own enjoyment"
        ],
        "9": [
            "I strive for personal success",
            "I enjoy being in charge"
        ],
        "10": [
            "I care about all humanity",
            "I help others selflessly"
        ],
        "11": [
            "I enjoy leadership roles",
            "I adapt my role as needed"
        ]
    }
    
    responses = {}
    trait_responses = {}

    # Create 2 columns for questions
    col1, col2 = st.columns(2)

    # Display questions in two columns
    for i, (question, trait) in enumerate(questions, 1):
        current_col = col1 if i % 2 == 1 else col2
        
        with current_col:
            with st.container():
                st.markdown(f'<div class="question-column">'
                           f'<span class="question-number">Q{i}.</span> {question}'
                           f'</div>', unsafe_allow_html=True)
                
                # Get response (1-5 scale)
                response = st.select_slider(
                    f"Response to Q{i}",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    key=f"Q{i}",
                    label_visibility="collapsed"
                )
                
                responses[f"Q{i}"] = response
                if trait not in trait_responses:
                    trait_responses[trait] = []
                trait_responses[trait].append(response)

    if st.button("Submit Personality Assessment", use_container_width=True):
        # Calculate normalized scores (0-1 range)
        trait_scores = {}
        for trait, values in trait_responses.items():
            # Convert from 1-5 scale to 0-1 scale
            normalized_values = [(x - 1) / 4 for x in values]
            trait_scores[trait] = np.mean(normalized_values)
        
        # Display results
        st.success("Assessment complete!")
        st.subheader("Your Personality Profile")
        
        # Display in 3 columns with progress bars
        cols = st.columns(3)
        traits = sorted(trait_scores.keys())
        
        for i, trait in enumerate(traits):
            with cols[i % 3]:
                score = trait_scores[trait]
                st.markdown(f"**{trait}**")
                # Display progress bar with percentage
                st.progress(score, text=f"{score:.0%}")
                # Interpretation text
                st.caption(get_interpretation(trait, score))
        
        return trait_scores

def get_interpretation(trait, score):
    """Helper function to provide interpretation of normalized scores (0-1)"""
    if score < 0.3:
        level = "Low"
    elif score < 0.7:
        level = "Moderate"
    else:
        level = "High"
    
    if trait == "Emotional Range":
        return f"{level} emotional reactivity"
    elif trait == "Openness to Change":
        return f"{level} preference for change"
    elif trait == "Role":
        return f"{level} role flexibility"
    else:
        return f"{level} {trait.lower()}"