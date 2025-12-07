import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="VoiceWeave",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
:root {
    --primary: #2563eb;
    --primary-dark: #1e40af;
    --primary-light: #eff6ff;
    --text: #1f2937;
    --text-light: #6b7280;
    --bg-white: #ffffff;
    --bg-light: #f9fafb;
    --border: #e5e7eb;
    --green: #10b981;
    --yellow: #f59e0b;
    --red: #ef4444;
}

* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: var(--text);
}

body {
    background-color: var(--bg-light);
}

/* Streamlit overrides */
.stApp {
    background-color: var(--bg-light);
}

.stMarkdown {
    color: var(--text);
}

/* Buttons */
.stButton > button {
    background-color: var(--primary);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    transition: all 0.2s ease;
    width: 100%;
}

.stButton > button:hover {
    background-color: var(--primary-dark);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    transform: translateY(-1px);
}

/* Text input */
.stTextArea > textarea,
.stTextInput > input {
    border-color: var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Courier New', monospace !important;
}

.stTextArea > textarea:focus,
.stTextInput > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-light) !important;
}

/* Cards/Metrics */
.stMetric {
    background-color: var(--bg-white);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

/* Sidebar */
.stSidebar {
    background-color: var(--bg-white);
    border-right: 1px solid var(--border);
}

.stSidebar .stMarkdown {
    color: var(--text);
}

/* Status badges */
.status-online {
    color: var(--green);
    font-weight: 600;
}

.status-offline {
    color: var(--red);
    font-weight: 600;
}

/* Alert boxes */
.alert {
    border-radius: 8px;
    padding: 16px;
    margin: 16px 0;
    border-left: 4px solid;
}

.alert-error {
    background-color: #fee2e2;
    border-color: var(--red);
    color: #7f1d1d;
}

.alert-warning {
    background-color: #fef3c7;
    border-color: var(--yellow);
    color: #78350f;
}

.alert-info {
    background-color: var(--primary-light);
    border-color: var(--primary);
    color: #1e3a8a;
}

.alert-success {
    background-color: #d1fae5;
    border-color: var(--green);
    color: #065f46;
}

/* Heatmap */
.heatmap-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 20px 0;
}

.heatmap-cell {
    width: 50px;
    height: 50px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 12px;
    border: 1px solid rgba(0,0,0,0.1);
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Recommendation */
.recommendation {
    background-color: var(--primary-light);
    border-left: 4px solid var(--primary);
    padding: 16px;
    border-radius: 8px;
    margin: 12px 0;
}

.recommendation-title {
    font-weight: 600;
    color: var(--primary-dark);
    margin-bottom: 8px;
}

.recommendation-text {
    color: var(--text);
    margin: 0;
    line-height: 1.6;
}

.recommendation-meta {
    color: var(--text-light);
    font-size: 13px;
    margin-top: 8px;
}

/* Section titles */
h1 {
    color: var(--text);
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 8px;
    border-bottom: 3px solid var(--primary);
    padding-bottom: 12px;
}

h2 {
    color: var(--text);
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 16px;
    border-bottom: 2px solid var(--primary);
    padding-bottom: 8px;
}

h3 {
    color: var(--text);
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Subtitle */
.subtitle {
    color: var(--text-light);
    font-size: 16px;
    margin: -12px 0 20px 0;
}

/* Data table */
.stDataFrame {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# Backend configuration
BACKEND_URL = "http://localhost:5000"

def check_backend():
    """Check if backend is available"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return response.status_code == 200, response.json()
    except:
        return False, None

# Sidebar Navigation
with st.sidebar:
    st.title("VoiceWeave")
    st.markdown("**Mechanistic Analysis of Voice Suppression**")
    st.divider()
    
    # Backend status
    backend_ok, backend_info = check_backend()
    if backend_ok:
        st.markdown('<p class="status-online">✓ Backend: Online</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-offline">✗ Backend: Offline</p>', unsafe_allow_html=True)
        st.markdown("""
<div class="alert alert-error">
<strong>Backend Issue</strong><br>
Make sure Flask is running:<br>
de>python backend.py</code>
</div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Navigation
    page = st.radio(
        "Navigate",
        ["About", "How to Use", "Analyze", "Research"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Project info
    st.markdown("""
**Project Information**
- Version: 1.0
- Type: Research Prototype
- Status: Active
- Models: RF (79% AUC) + LSTM (69% AUC)
- Data: 38,666 utterances
    """)

# PAGE 1: ABOUT
if page == "About":
    st.markdown("# VoiceWeave")
    st.markdown("### Mechanistic Detection and Intervention for Voice Suppression in Dialogue")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Detection Accuracy", "79%", "AUC-ROC")
    with col2:
        st.metric("Prediction Accuracy", "69%", "AUC-ROC")
    with col3:
        st.metric("Intervention Success", "59%", "Flip Rate")
    
    st.divider()
    
    st.markdown("## What is VoiceWeave?")
    st.markdown("""
VoiceWeave is a research system for analyzing conversational voice suppression—the phenomenon where certain speakers are systematically marginalized or overlooked in multi-party dialogues.

Unlike sentiment analysis or toxicity detection, VoiceWeave models suppression as a **structural property** of dialogue. It examines turn timing, speaker activity patterns, and participation dynamics to identify who is being heard and who is being overlooked.
    """)
    
    st.markdown("## Core Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Detection")
        st.markdown("""
Trained Random Forest identifies suppressed turns with 79% accuracy. Learned from 38,666 labeled utterances spanning Chinese TV drama and English conversations.
        """)
        
        st.markdown("### Mechanistic Interpretability")
        st.markdown("""
Explains predictions using SHAP values and attention mechanisms. See exactly which features drive each prediction and why suppression occurs.
        """)
    
    with col2:
        st.markdown("### Prediction")
        st.markdown("""
LSTM with attention forecasts future suppression. Identifies dialogue patterns that precede marginalization. 69% AUC.
        """)
        
        st.markdown("### Intervention Design")
        st.markdown("""
Proposes structural changes (turn timing, participation patterns) to reduce suppression. 59% of interventions successfully flip suppressed turns.
        """)
    
    st.divider()
    
    st.markdown("""
<div class="alert alert-warning">
<strong>Research Prototype</strong><br>
This system is a research demonstration. Current implementation requires pre-labeled transcripts and achieves ~73-77% accuracy on new domains. Use as analytical guidance, not ground truth.
</div>
    """, unsafe_allow_html=True)

# PAGE 2: HOW TO USE
elif page == "How to Use":
    st.markdown("# How to Use VoiceWeave")
    
    st.markdown("## Step 1: Prepare Your Transcript")
    st.markdown("""
VoiceWeave accepts simple dialogue format:

**Format:** `speaker_name|text` (one turn per line)

**Example:**
