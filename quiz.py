import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os
import torch
import torch.nn as nn
import torch.optim as optim

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
        .reset-btn {
            margin-top: 20px;
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
    
    # Initialize session state for persistent storage
    if 'personality_scores' not in st.session_state:
        st.session_state.personality_scores = None
    if 'responses' not in st.session_state:
        st.session_state.responses = {}

    # Reset function
    def reset_results():
        st.session_state.personality_scores = None
        st.session_state.responses = {}
        st.rerun()

    # Display reset button if results exist
    if st.session_state.personality_scores:
        st.button("Reset Assessment Results", on_click=reset_results, key="reset_btn", 
                 help="Click to clear all assessment results", use_container_width=True)

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
                    
                    # Get response (1-5 scale), using stored value if exists
                    default_value = st.session_state.responses.get(f"Q{question_count}", 3)
                    response = st.select_slider(
                        f"Response to Q{question_count}",
                        options=[1, 2, 3, 4, 5],
                        value=default_value,
                        key=f"Q{question_count}_slider",
                        label_visibility="collapsed"
                    )
                    
                    # Store response in session state
                    st.session_state.responses[f"Q{question_count}"] = response

    if st.button("Submit Personality Assessment", use_container_width=True):
        # Calculate normalized scores (0-1 range)
        trait_scores = {}
        trait_responses = {}
        
        # Organize responses by trait
        question_count = 0
        for trait, trait_questions in questions.items():
            trait_responses[trait] = []
            for _ in trait_questions:
                question_count += 1
                trait_responses[trait].append(st.session_state.responses[f"Q{question_count}"])
        
        # Calculate scores
        for trait, values in trait_responses.items():
            normalized_values = [(x - 1) / 4 for x in values]
            trait_scores[trait] = np.mean(normalized_values)
        
        # Store in session state
        st.session_state.personality_scores = trait_scores
        st.success("Assessment complete! Results are now saved.")
        st.rerun()

    # Display results if available
    if st.session_state.personality_scores:
        st.subheader("Your Personality Profile")
        
        # Display in 3 columns with progress bars
        cols = st.columns(3)
        traits = sorted(st.session_state.personality_scores.keys())
        
        for i, trait in enumerate(traits):
            with cols[i % 3]:
                score = st.session_state.personality_scores[trait]
                st.markdown(f"**{trait}**")
                st.progress(score, text=f"{score:.0%}")
                st.caption(get_interpretation(trait, score))
        
        # Add another reset button at bottom
        st.button("Reset Assessment Results", on_click=reset_results, key="reset_btn_bottom", 
                 help="Click to clear all assessment results", use_container_width=True)

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