# Performance Optimization Functions for ML Playground
import streamlit as st
import requests
from functools import lru_cache
import time

# Optimized API Configuration
API_TIMEOUT_SHORT = 10  # For quick requests like fetching algorithms
API_TIMEOUT_LONG = 60   # For training operations

def check_api_health():
    """Check if API is healthy before using cached data"""
    try:
        response = requests.get(f"{st.session_state.get('api_base_url', 'http://localhost:8000')}/health", 
                              timeout=5)
        return response.status_code == 200
    except Exception:
        return False

@st.cache_data(ttl=300)  # Cache for 5 minutes
def _get_algorithms_from_api():
    """Internal function to fetch algorithms from API"""
    try:
        response = requests.get(f"{st.session_state.get('api_base_url', 'http://localhost:8000')}/algorithms", 
                              timeout=API_TIMEOUT_SHORT)
        if response.status_code == 200:
            algorithms = response.json()
            if algorithms:  # Only cache non-empty responses
                return algorithms
        return None
    except Exception:
        return None

def get_algorithms_cached():
    """Smart cached version of get_algorithms that handles API downtime"""
    # First check if API is healthy
    if not check_api_health():
        # API is down - clear any stale cache and return empty
        _get_algorithms_from_api.clear()
        return []
    
    # API is healthy - try to get cached data
    algorithms = _get_algorithms_from_api()
    if algorithms is None:
        # Cache miss or API error - clear cache and return empty
        _get_algorithms_from_api.clear()
        return []
    
    return algorithms

@st.cache_data(ttl=600)  # Cache for 10 minutes  
def _get_datasets_from_api():
    """Internal function to fetch datasets from API"""
    try:
        response = requests.get(f"{st.session_state.get('api_base_url', 'http://localhost:8000')}/datasets",
                              timeout=API_TIMEOUT_SHORT)
        if response.status_code == 200:
            datasets = response.json()
            if datasets and datasets.get("builtin_datasets"):  # Only cache valid responses
                return datasets
        return None
    except Exception:
        return None

def get_datasets_cached():
    """Smart cached version of get_datasets that handles API downtime"""
    # First check if API is healthy
    if not check_api_health():
        # API is down - clear any stale cache and return empty
        _get_datasets_from_api.clear()
        return {"builtin_datasets": []}
    
    # API is healthy - try to get cached data
    datasets = _get_datasets_from_api()
    if datasets is None:
        # Cache miss or API error - clear cache and return empty
        _get_datasets_from_api.clear()
        return {"builtin_datasets": []}
    
    return datasets

def clear_api_cache():
    """Clear all API-related caches - useful when API comes back online"""
    _get_algorithms_from_api.clear()
    _get_datasets_from_api.clear()
    st.success("🔄 Cache cleared! API data will be refreshed.")

def refresh_api_data():
    """Force refresh API data by clearing cache and checking health"""
    clear_api_cache()
    if check_api_health():
        st.success("✅ API is healthy! Data will be refreshed on next request.")
        return True
    else:
        st.error("❌ API is still not accessible. Please check if the backend is running.")
        return False

@lru_cache(maxsize=32)
def get_minified_css():
    """Get minified CSS for faster loading"""
    return """
    <style>
    .main{font-family:'Inter',sans-serif}
    .main-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:2.5rem;border-radius:15px;color:white;text-align:center;margin-bottom:2rem;box-shadow:0 10px 30px rgba(102,126,234,0.3)}
    .main-header h1{font-size:3rem;font-weight:700;margin:0;text-shadow:2px 2px 4px rgba(0,0,0,0.3)}
    .algorithm-card{background:white;padding:1.5rem;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.08);border-left:4px solid #667eea;margin-bottom:1rem;transition:transform 0.2s ease;border:1px solid #f0f0f0}
    .algorithm-card:hover{transform:translateY(-2px);box-shadow:0 8px 30px rgba(0,0,0,0.12)}
    .metric-card{background:#f8f9fa;padding:1.5rem;border-radius:10px;text-align:center;border:1px solid #e9ecef}
    .metric-value{font-size:2rem;font-weight:600;color:#2c3e50;margin:0}
    .metric-label{color:#6c757d;font-size:0.9rem;margin:0}
    .feature-highlight{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:1.5rem;border-radius:10px;margin:1rem 0}
    .section-header{background:#f8f9fa;padding:1rem;border-radius:8px;border-left:4px solid #667eea;margin:1rem 0}
    .status-success{color:#28a745;font-weight:500}
    .status-error{color:#dc3545;font-weight:500}
    .stButton > button{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;border:none;border-radius:8px;padding:0.5rem 1.5rem;font-weight:600;transition:all 0.2s ease}
    .stButton > button:hover{transform:translateY(-1px);box-shadow:0 4px 15px rgba(102,126,234,0.4)}
    </style>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    """

def optimize_session_state():
    """Clean up session state to improve performance"""
    # Limit experiment history to last 20 items
    if 'experiment_history' in st.session_state:
        if len(st.session_state.experiment_history) > 20:
            st.session_state.experiment_history = st.session_state.experiment_history[:20]
    
    # Clear old cached data periodically
    if 'last_cleanup' not in st.session_state:
        st.session_state.last_cleanup = time.time()
    
    # Clean up every 5 minutes
    if time.time() - st.session_state.last_cleanup > 300:
        if 'cached_algorithms' in st.session_state:
            del st.session_state.cached_algorithms
        if 'cached_datasets' in st.session_state:
            del st.session_state.cached_datasets
        st.session_state.last_cleanup = time.time()

def train_model_optimized(algorithm_id, hyperparameters, dataset_config, compare_sklearn=True, 
                         dataset_source="generated", builtin_dataset="diabetes", uploaded_data=None):
    """Optimized train_model function with better error handling and timeout"""
    try:
        payload = {
            "algorithm_id": algorithm_id,
            "hyperparameters": hyperparameters,
            "dataset_config": dataset_config,
            "compare_sklearn": compare_sklearn,
            "dataset_source": dataset_source,
            "builtin_dataset": builtin_dataset,
            "uploaded_data": uploaded_data
        }
        
        # Show progress while training
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Preparing training data...")
        progress_bar.progress(25)
        
        # Make the request with proper timeout
        api_url = st.session_state.get('api_base_url', 'http://localhost:8000')
        response = requests.post(f"{api_url}/train", json=payload, timeout=API_TIMEOUT_LONG)
        
        progress_bar.progress(75)
        status_text.text("Processing results...")
        
        if response.status_code == 200:
            progress_bar.progress(100)
            status_text.text("Training completed!")
            time.sleep(0.5)  # Brief pause to show completion
            progress_bar.empty()
            status_text.empty()
            return response.json()
        else:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Training failed: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏱️ Training timed out. Try reducing dataset size or iterations.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Could not connect to the backend API. Make sure the server is running.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None
    finally:
        # Clean up progress indicators
        try:
            progress_bar.empty()
            status_text.empty()
        except:
            pass

@st.cache_data
def create_metric_card_optimized(title, value, delta=None, delta_color="normal"):
    """Cached metric card creation for better performance"""
    delta_html = ""
    if delta is not None:
        color = "#28a745" if delta_color == "normal" and delta > 0 else "#dc3545" if delta < 0 else "#28a745"
        arrow = "↗" if delta > 0 else "↘" if delta < 0 else "→"
        delta_html = f'<div style="color: {color}; font-size: 0.9rem; margin-top: 0.25rem;">{arrow} {delta:+.3f}</div>'
    
    # Format value properly
    formatted_value = f"{value:.3f}" if isinstance(value, float) else str(value)
    
    return f"""
    <div class="metric-card">
        <div class="metric-value">{formatted_value}</div>
        <div class="metric-label">{title}</div>
        {delta_html}
    </div>
    """

def lazy_load_visualizations():
    """Lazy loading wrapper for heavy visualizations"""
    if 'show_visualizations' not in st.session_state:
        st.session_state.show_visualizations = False
    
    if st.button("📊 Show Visualizations"):
        st.session_state.show_visualizations = True
    
    return st.session_state.show_visualizations

def optimize_plotly_config():
    """Return optimized Plotly configuration"""
    return {
        'displayModeBar': False,
        'responsive': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'zoom2d']
    }
