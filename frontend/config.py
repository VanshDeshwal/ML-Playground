# Configuration and constants
import os
import streamlit as st

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
PAGE_CONFIG = {
    "page_title": "ML Playground",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Session state initialization
def initialize_session_state():
    """Initialize all session state variables"""
    if 'experiment_history' not in st.session_state:
        st.session_state.experiment_history = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'
    if 'selected_algorithm' not in st.session_state:
        st.session_state.selected_algorithm = None
    if 'api_base_url' not in st.session_state:
        st.session_state.api_base_url = API_BASE_URL

# Performance settings
CACHE_TTL_SHORT = 300   # 5 minutes for algorithms/datasets
CACHE_TTL_LONG = 3600   # 1 hour for CSS and static content
MAX_EXPERIMENT_HISTORY = 20  # Limit history to prevent memory bloat
