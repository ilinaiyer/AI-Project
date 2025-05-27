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
        .question-column {
            padding: 10px;
            margin-bottom: 15px;
        }
        .question-number {
            font-weight: bold;
            color: #0d6efd;
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
    
    if 'personality_scores' not in st.session_state:
        st.session_state.personality_scores = None
    responses = {}

    trait_responses = {}

    # Create 2 columns for questions
    col1, col2 = st.columns(2)
    question_count = 0

    # Display questions in two columns
    for trait, trait_questions in questions.items():
        for i, question in enumerate(trait_questions):
            question_count += 1
            current_col = col1 if question_count % 2 == 1 else col2
            
            with current_col:
                with st.container():
                    st.markdown(f'<div class="question-column">'
                               f'<span class="question-number">Q{question_count}.</span> {question}'
                               f'</div>', unsafe_allow_html=True)
                    
                    # Get response (1-5 scale)
                    response = st.select_slider(
                        f"Response to Q{question_count}",
                        options=[1, 2, 3, 4, 5],
                        value=3,
                        key=f"Q{question_count}",
                        label_visibility="collapsed"
                    )
                    
                    responses[f"Q{question_count}"] = response
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

        st.session_state.personality_scores = trait_scores
        st.success("Assessment complete!")
        st.subheader("Your Personality Profile")
        
        if st.session_state.personality_scores:
            st.subheader("Your Results")
            for trait, score in st.session_state.personality_scores.items():
                st.write(f"{trait}: {score:.2f}")
                st.progress(score)
                
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