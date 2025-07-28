# API communication utilities
import streamlit as st
import requests
from datetime import datetime
from config import API_BASE_URL

def add_experiment_to_history(algorithm_name, result, timestamp):
    """Add experiment to session history with performance optimization"""
    experiment = {
        'timestamp': timestamp,
        'algorithm': algorithm_name,
        'accuracy': result.get('custom_score', 0),
        'status': 'Success' if result else 'Failed'
    }
    
    # Add to beginning and limit history size
    st.session_state.experiment_history.insert(0, experiment)
    if len(st.session_state.experiment_history) > 20:
        st.session_state.experiment_history = st.session_state.experiment_history[:20]

def check_backend_health():
    """Check if backend API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

@st.cache_data(ttl=60)  # Cache health status for 1 minute
def get_backend_status():
    """Get cached backend status"""
    if check_backend_health():
        return {"status": "healthy", "message": "Backend API is running"}
    else:
        return {"status": "error", "message": "Backend API is not accessible"}

def safe_api_call(url, method="GET", **kwargs):
    """Safe API call wrapper with error handling"""
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=kwargs.get('timeout', 10))
        elif method.upper() == "POST":
            response = requests.post(url, timeout=kwargs.get('timeout', 30), **kwargs)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Could not connect to backend. Make sure the server is running.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None
