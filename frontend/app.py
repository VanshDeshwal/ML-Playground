import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import time
import os
from functools import lru_cache

# Import performance improvements
from performance_improvements import (
    get_algorithms_cached, get_datasets_cached, get_minified_css,
    optimize_session_state, train_model_optimized, create_metric_card_optimized,
    lazy_load_visualizations, optimize_plotly_config
)

# Configure page
st.set_page_config(
    page_title="ML Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state and apply optimizations
optimize_session_state()

# Load optimized CSS
st.markdown(get_minified_css(), unsafe_allow_html=True)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Initialize session state
if 'experiment_history' not in st.session_state:
    st.session_state.experiment_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'
if 'selected_algorithm' not in st.session_state:
    st.session_state.selected_algorithm = None

def add_experiment_to_history(algorithm_name, result, timestamp):
    """Add experiment to session history"""
    experiment = {
        'timestamp': timestamp,
        'algorithm': algorithm_name,
        'accuracy': result.get('metrics', {}).get('accuracy', result.get('metrics', {}).get('r2_score', 0)),
        'dataset': result.get('dataset_info', {}).get('type', 'Unknown'),
        'status': 'Success' if result.get('success', True) else 'Failed'
    }
    st.session_state.experiment_history.insert(0, experiment)
    if len(st.session_state.experiment_history) > 50:  # Keep last 50 experiments
        st.session_state.experiment_history = st.session_state.experiment_history[:50]

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a metric card with optional delta"""
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

def create_algorithm_card(algo, is_selected=False):
    """Create an algorithm card"""
    icons = {
        'regression': '📈',
        'classification': '🎯', 
        'clustering': '🔍'
    }
    
    border_color = "#667eea" if is_selected else "#e9ecef"
    background = "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)" if is_selected else "white"
    
    return f"""
    <div class="algorithm-card" style="border-left-color: {border_color}; background: {background};">
        <div class="algorithm-icon">{icons.get(algo['type'], '🤖')}</div>
        <h4 style="margin: 0.5rem 0; color: #495057;">{algo['name']}</h4>
        <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">{algo['description']}</p>
        <div style="margin-top: 1rem;">
            <span class="status-badge status-success">{algo['type'].title()}</span>
        </div>
    </div>
    """

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_algorithms():
    """Fetch available algorithms from API with caching"""
    try:
        response = requests.get(f"{API_BASE_URL}/algorithms", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch algorithms")
            return []
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Make sure the FastAPI server is running.")
        return []
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please check your connection.")
        return []

def train_model(algorithm_id, hyperparameters, dataset_config, compare_sklearn=True, dataset_source="generated", builtin_dataset="diabetes", uploaded_data=None):
    """Train a model via API"""
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
        response = requests.post(f"{API_BASE_URL}/train", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Training failed: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API.")
        return None

def get_datasets():
    """Fetch available datasets from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/datasets")
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch datasets")
            return {"builtin_datasets": []}
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API.")
        return {"builtin_datasets": []}

def upload_dataset(uploaded_file):
    """Upload a dataset via API"""
    try:
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API.")
        return None

def main():
    """Main application with modern UI"""
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 2rem;">
            <h2 style="margin: 0;">🤖 ML Playground</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Interactive Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation Menu
        page = st.selectbox(
            "Navigate",
            ["🏠 Home", "🔬 Algorithm Explorer", "📊 Dashboard", "📈 Comparison Lab", "📚 Documentation"],
            key="navigation"
        )
        
        st.session_state.current_page = page.split(" ", 1)[1]  # Remove emoji
        
        # Quick Stats in Sidebar
        if st.session_state.experiment_history:
            st.markdown("### 📊 Quick Stats")
            total_experiments = len(st.session_state.experiment_history)
            successful_experiments = len([e for e in st.session_state.experiment_history if e['status'] == 'Success'])
            success_rate = (successful_experiments / total_experiments) * 100 if total_experiments > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Experiments", total_experiments)
            with col2:
                st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # API Management
        st.markdown("---")
        st.markdown("### 🔄 API Status")
        from performance_improvements import check_api_health, refresh_api_data
        
        if check_api_health():
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")
            
        if st.button("🔄 Refresh API Data", help="Clear cache and refresh data from backend"):
            refresh_api_data()
            st.rerun()
    
    # Main Content Area
    if st.session_state.current_page == "Home":
        show_home_page()
    elif st.session_state.current_page == "Algorithm Explorer":
        show_algorithm_explorer()
    elif st.session_state.current_page == "Dashboard":
        show_dashboard()
    elif st.session_state.current_page == "Comparison Lab":
        show_comparison_lab()
    elif st.session_state.current_page == "Documentation":
        show_documentation()

def show_home_page():
    """Modern home page with hero section and algorithm gallery"""
    # API Status Check
    from performance_improvements import check_api_health
    if not check_api_health():
        st.warning("⚠️ Backend API is not running. Please start the backend server to access algorithms and training features.")
        st.info("💡 **Tip:** Start the backend with `python backend/main.py` or use the 'Refresh API Data' button in the sidebar when it's ready.")
        return
    
    # Hero Section
    st.markdown("""
    <div class="main-header">
        <h1>🤖 ML Playground</h1>
        <p>Explore, Learn, and Master Machine Learning Algorithms</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Action Buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🚀 Start Learning", use_container_width=True):
            st.session_state.current_page = "Algorithm Explorer"
            st.rerun()
    with col2:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.current_page = "Dashboard"
            st.rerun()
    with col3:
        if st.button("📈 Compare Models", use_container_width=True):
            st.session_state.current_page = "Comparison Lab"
            st.rerun()
    with col4:
        if st.button("📚 Documentation", use_container_width=True):
            st.session_state.current_page = "Documentation"
            st.rerun()
    
    st.markdown("---")
    
    # Algorithm Gallery
    algorithms = get_algorithms_cached()
    if algorithms:
        st.markdown("""
        <div class="section-header">
            <h3>🎯 Available Algorithms</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Group algorithms by type
        algo_groups = {}
        for algo in algorithms:
            algo_type = algo['type']
            if algo_type not in algo_groups:
                algo_groups[algo_type] = []
            algo_groups[algo_type].append(algo)
        
        tabs = st.tabs([f"{key.title()} ({len(algos)})" for key, algos in algo_groups.items()])
        
        for i, (algo_type, algos) in enumerate(algo_groups.items()):
            with tabs[i]:
                cols = st.columns(min(3, len(algos)))
                for j, algo in enumerate(algos):
                    with cols[j % 3]:
                        if st.button(f"Try {algo['name']}", key=f"try_{algo['id']}", use_container_width=True):
                            st.session_state.selected_algorithm = algo['id']
                            st.session_state.current_page = "Algorithm Explorer"
                            st.rerun()
                        st.markdown(create_algorithm_card(algo), unsafe_allow_html=True)
    
    # Recent Activity
    if st.session_state.experiment_history:
        st.markdown("""
        <div class="section-header">
            <h3>🕒 Recent Activity</h3>
        </div>
        """, unsafe_allow_html=True)
        
        recent_experiments = st.session_state.experiment_history[:5]
        for exp in recent_experiments:
            status_class = "status-success" if exp['status'] == 'Success' else "status-error"
            st.markdown(f"""
            <div class="experiment-row">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{exp['algorithm']}</strong> • {exp['dataset']} dataset
                        <div style="font-size: 0.8rem; color: #6c757d;">{exp['timestamp']}</div>
                    </div>
                    <div>
                        <span class="status-badge {status_class}">{exp['status']}</span>
                        <div style="font-size: 0.9rem; margin-top: 0.25rem;">Score: {exp['accuracy']:.3f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Feature Highlights
    st.markdown("""
    <div class="feature-highlight">
        <h4 style="margin-top: 0; color: #495057;">🎯 What makes ML Playground special?</h4>
        <ul style="color: #6c757d;">
            <li><strong>Interactive Learning:</strong> Hands-on experience with real algorithms</li>
            <li><strong>Custom Implementations:</strong> Understand how algorithms work under the hood</li>
            <li><strong>Scikit-learn Comparison:</strong> Compare your understanding with industry standards</li>
            <li><strong>Rich Visualizations:</strong> See algorithms in action with beautiful charts</li>
            <li><strong>Multiple Datasets:</strong> Test on built-in, generated, or custom datasets</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def show_dashboard():
    """Enhanced dashboard with experiment tracking"""
    st.markdown("""
    <div class="section-header">
        <h3>📊 ML Playground Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.experiment_history:
        st.info("🚀 No experiments yet! Head over to the Algorithm Explorer to start your ML journey.")
        if st.button("🔬 Start Exploring", use_container_width=True):
            st.session_state.current_page = "Algorithm Explorer"
            st.rerun()
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_exp = len(st.session_state.experiment_history)
    successful_exp = len([e for e in st.session_state.experiment_history if e['status'] == 'Success'])
    avg_accuracy = np.mean([e['accuracy'] for e in st.session_state.experiment_history if e['status'] == 'Success']) if successful_exp > 0 else 0
    unique_algos = len(set([e['algorithm'] for e in st.session_state.experiment_history]))
    
    with col1:
        st.markdown(create_metric_card("Total Experiments", total_exp), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Success Rate", f"{(successful_exp/total_exp)*100:.1f}%"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("Avg. Score", avg_accuracy), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card("Algorithms Tried", unique_algos), unsafe_allow_html=True)
    
    # Experiment timeline
    st.markdown("### 📈 Performance Timeline")
    if successful_exp > 0:
        df = pd.DataFrame([e for e in st.session_state.experiment_history if e['status'] == 'Success'])
        df['experiment_id'] = range(len(df), 0, -1)
        
        fig = px.line(df, x='experiment_id', y='accuracy', color='algorithm',
                     title="Model Performance Over Time",
                     labels={'experiment_id': 'Experiment Number', 'accuracy': 'Score'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, config=optimize_plotly_config())
    
    # Algorithm performance comparison
    st.markdown("### 🏆 Algorithm Performance Comparison")
    if successful_exp > 0:
        algo_performance = {}
        for exp in st.session_state.experiment_history:
            if exp['status'] == 'Success':
                if exp['algorithm'] not in algo_performance:
                    algo_performance[exp['algorithm']] = []
                algo_performance[exp['algorithm']].append(exp['accuracy'])
        
        algo_avg = {algo: np.mean(scores) for algo, scores in algo_performance.items()}
        
        fig = px.bar(x=list(algo_avg.keys()), y=list(algo_avg.values()),
                    title="Average Performance by Algorithm",
                    labels={'x': 'Algorithm', 'y': 'Average Score'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, config=optimize_plotly_config())
    
    # Detailed experiment history
    st.markdown("### 📋 Experiment History")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.experiment_history = []
            st.rerun()
    
    # Create a detailed table
    df_display = pd.DataFrame(st.session_state.experiment_history)
    if not df_display.empty:
        df_display = df_display.round(3)
        st.dataframe(df_display, use_container_width=True, height=400)

def show_comparison_lab():
    """Comparison lab for side-by-side algorithm comparison"""
    st.markdown("""
    <div class="section-header">
        <h3>📈 Comparison Lab</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-highlight">
        <h4 style="margin-top: 0;">🔬 Algorithm Comparison Lab</h4>
        <p style="margin-bottom: 0;">Compare multiple algorithms side-by-side on the same dataset to understand their relative performance and characteristics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    algorithms = get_algorithms_cached()
    if not algorithms:
        st.error("Unable to load algorithms. Please check the backend connection.")
        return
    
    # Algorithm selection for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🥊 Algorithm 1")
        algo1 = st.selectbox("Select first algorithm", algorithms, format_func=lambda x: x['name'], key="comp_algo1")
        
    with col2:
        st.markdown("#### 🥊 Algorithm 2") 
        algo2 = st.selectbox("Select second algorithm", algorithms, format_func=lambda x: x['name'], key="comp_algo2")
    
    if algo1['type'] != algo2['type']:
        st.warning("⚠️ Selected algorithms are of different types. Comparison may not be meaningful.")
    
    # Dataset configuration
    st.markdown("#### 📊 Dataset Configuration")
    dataset_source = st.radio("Dataset Source", ["Generated", "Built-in"], horizontal=True, key="comp_dataset")
    
    if dataset_source == "Generated":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_samples = st.slider("Samples", 100, 1000, 500, key="comp_samples")
        with col2:
            n_features = st.slider("Features", 2, 10, 5, key="comp_features")
        with col3:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, key="comp_test_size")
        
        dataset_config = {
            "n_samples": n_samples,
            "n_features": n_features,
            "noise": 0.1,
            "test_size": test_size,
            "random_state": 42
        }
        dataset_source_api = "generated"
        builtin_dataset = None
    else:
        datasets_info = get_datasets_cached()
        available_datasets = [d for d in datasets_info["builtin_datasets"] if d['type'] == algo1['type']]
        
        if available_datasets:
            selected_dataset = st.selectbox("Select Dataset", available_datasets, format_func=lambda x: x['name'], key="comp_builtin")
            builtin_dataset = selected_dataset['id']
            dataset_config = {"test_size": 0.2}
            dataset_source_api = "builtin"
        else:
            st.warning("No compatible datasets found. Using generated data.")
            dataset_config = {"n_samples": 500, "n_features": 5, "noise": 0.1, "test_size": 0.2, "random_state": 42}
            dataset_source_api = "generated"
            builtin_dataset = None
    
    # Run comparison
    if st.button("🚀 Run Comparison", use_container_width=True):
        with st.spinner("Training algorithms..."):
            # Train both algorithms with same data
            results = {}
            
            for i, algo in enumerate([algo1, algo2], 1):
                # Use default hyperparameters
                hyperparameters = {param: config['default'] for param, config in algo['hyperparameters'].items()}
                
                result = train_model_optimized(
                    algo['id'],
                    hyperparameters,
                    dataset_config,
                    compare_sklearn=True,
                    dataset_source=dataset_source_api,
                    builtin_dataset=builtin_dataset
                )
                
                if result:
                    results[f"algo_{i}"] = {"result": result, "algo": algo}
                    # Add to history
                    add_experiment_to_history(
                        algo['name'],
                        result,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
            
            # Display comparison results
            if len(results) == 2:
                st.markdown("### 🏆 Comparison Results")
                
                col1, col2 = st.columns(2)
                
                for i, (key, data) in enumerate(results.items()):
                    result = data["result"]
                    algo = data["algo"]
                    
                    with col1 if i == 0 else col2:
                        st.markdown(f"#### {algo['name']}")
                        
                        # Display metrics
                        metrics = result.get('metrics', {})
                        sklearn_metrics = result.get('sklearn_comparison', {}).get('metrics', {})
                        
                        # Main metric based on algorithm type
                        if algo['type'] == 'regression':
                            main_metric = metrics.get('r2_score', 0)
                            sklearn_main = sklearn_metrics.get('r2_score', 0)
                            metric_name = "R² Score"
                        elif algo['type'] == 'classification':
                            main_metric = metrics.get('accuracy', 0)
                            sklearn_main = sklearn_metrics.get('accuracy', 0)
                            metric_name = "Accuracy"
                        else:  # clustering
                            main_metric = metrics.get('silhouette_score', 0)
                            sklearn_main = sklearn_metrics.get('silhouette_score', 0)
                            metric_name = "Silhouette Score"
                        
                        delta = main_metric - sklearn_main if sklearn_main else None
                        st.markdown(create_metric_card(f"Custom {metric_name}", main_metric, delta), unsafe_allow_html=True)
                        st.markdown(create_metric_card(f"Sklearn {metric_name}", sklearn_main), unsafe_allow_html=True)

def show_documentation():
    """Enhanced documentation page"""
    st.markdown("""
    <div class="section-header">
        <h3>📚 Documentation & Learning Resources</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Algorithm documentation tabs
    algorithms = get_algorithms_cached()
    if algorithms:
        tabs = st.tabs(["📖 Overview", "🧮 Algorithms", "📊 Datasets", "🎯 Tips & Best Practices"])
        
        with tabs[0]:
            st.markdown("""
            <div class="feature-highlight">
                <h4 style="margin-top: 0;">🎯 Welcome to ML Playground!</h4>
                <p>ML Playground is an interactive platform designed to help you understand machine learning algorithms through hands-on experimentation.</p>
            </div>
            
            ### 🚀 Getting Started
            1. **Choose an Algorithm**: Navigate to the Algorithm Explorer
            2. **Configure Parameters**: Adjust hyperparameters to see their effects
            3. **Select Data**: Use generated, built-in, or upload your own datasets
            4. **Train & Analyze**: Run the algorithm and explore the results
            5. **Compare**: See how your custom implementation compares with scikit-learn
            
            ### 🎨 Features
            - **Custom Implementations**: All algorithms implemented from scratch for educational purposes
            - **Interactive Visualizations**: Rich plots and charts to understand algorithm behavior
            - **Scikit-learn Comparison**: Compare with industry-standard implementations
            - **Multiple Datasets**: Variety of built-in datasets plus support for custom data
            - **Experiment Tracking**: Keep track of your experiments and progress
            """, unsafe_allow_html=True)
        
        with tabs[1]:
            st.markdown("### 🧮 Available Algorithms")
            
            algo_groups = {}
            for algo in algorithms:
                algo_type = algo['type']
                if algo_type not in algo_groups:
                    algo_groups[algo_type] = []
                algo_groups[algo_type].append(algo)
            
            for algo_type, algos in algo_groups.items():
                st.markdown(f"#### {algo_type.title()}")
                for algo in algos:
                    with st.expander(f"{algo['name']} - {algo['description']}"):
                        st.markdown(f"**Type**: {algo['type']}")
                        st.markdown("**Hyperparameters**:")
                        for param, config in algo['hyperparameters'].items():
                            st.markdown(f"- **{param}**: {config.get('description', 'No description available')}")
        
        with tabs[2]:
            st.markdown("### 📊 Available Datasets")
            datasets_info = get_datasets_cached()
            
            if datasets_info["builtin_datasets"]:
                for dataset in datasets_info["builtin_datasets"]:
                    with st.expander(f"{dataset['name']} ({dataset['type']})"):
                        st.markdown(f"**Description**: {dataset['description']}")
                        st.markdown(f"**Type**: {dataset['type']}")
            
            st.markdown("""
            ### 📁 Custom Datasets
            You can also upload your own CSV files:
            - Last column should be the target variable
            - Numerical data works best
            - Missing values will be handled automatically
            """)
        
        with tabs[3]:
            st.markdown("""
            ### 🎯 Tips & Best Practices
            
            #### 🔧 Hyperparameter Tuning
            - Start with default values and adjust gradually
            - For learning rates: try values like 0.001, 0.01, 0.1
            - For regularization: start small (0.01) and increase if overfitting
            
            #### 📊 Data Preparation
            - Ensure your data is clean and numerical
            - Consider feature scaling for distance-based algorithms
            - Use appropriate train/test splits (typically 70-80% training)
            
            #### 🧪 Experimentation
            - Try different algorithms on the same dataset
            - Compare custom implementations with scikit-learn
            - Document your findings in the experiment history
            
            #### 📈 Interpretation
            - Higher accuracy/R² is better for classification/regression
            - Look at confusion matrices for classification insights
            - Consider computational complexity for large datasets
            """, unsafe_allow_html=True)
    """Fetch available algorithms from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/algorithms")
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch algorithms")
            return []
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Make sure the FastAPI server is running.")
        return []

def train_model(algorithm_id, hyperparameters, dataset_config, compare_sklearn=True, dataset_source="generated", builtin_dataset="diabetes", uploaded_data=None):
    """Train a model via API"""
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
        response = requests.post(f"{API_BASE_URL}/train", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Training failed: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API.")
        return None

def get_datasets():
    """Fetch available datasets from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/datasets")
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch datasets")
            return {"builtin_datasets": []}
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API.")
        return {"builtin_datasets": []}

def upload_dataset(uploaded_file):
    """Upload a dataset via API"""
    try:
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{API_BASE_URL}/upload_dataset", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API.")
        return None

def main():
    st.title("🤖 ML Playground")
    st.markdown("### Explore Machine Learning Algorithms from Scratch")
    
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Check if session state page is different from selectbox
    page_options = ["Home", "Algorithm Explorer", "About"]
    
    # Set initial index based on session state
    if st.session_state.page in page_options:
        initial_index = page_options.index(st.session_state.page)
    else:
        initial_index = 0
    
    page = st.sidebar.selectbox(
        "Select Page",
        page_options,
        index=initial_index,
        key="page_selector"
    )
    
    # Update session state if page changed via selectbox
    if page != st.session_state.page:
        st.session_state.page = page
    
    # Route to appropriate page
    if st.session_state.page == "Home":
        show_home_page()
    elif st.session_state.page == "Algorithm Explorer":
        show_algorithm_explorer()
    elif st.session_state.page == "About":
        show_about_page()

def show_home_page():
    """Display the home page"""
    st.header("Welcome to ML Playground! 🎯")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **ML Playground** is an interactive platform to explore and experiment with machine learning algorithms 
        implemented from scratch. Here you can:
        
        - 🔬 **Experiment** with different ML algorithms
        - ⚙️ **Customize** hyperparameters and see their effects
        - 📊 **Visualize** training processes and results
        - 📈 **Compare** performance metrics
        - 🎓 **Learn** how algorithms work under the hood
        """)
        
        st.subheader("Available Algorithms")
        
        # Fetch and display algorithms
        algorithms = get_algorithms_cached()
        if algorithms:
            for algo in algorithms:
                with st.expander(f"🔧 {algo['name']}"):
                    st.write(f"**Type:** {algo['type'].title()}")
                    st.write(f"**Description:** {algo['description']}")
                    
                    # Show hyperparameters
                    st.write("**Hyperparameters:**")
                    for param, config in algo['hyperparameters'].items():
                        st.write(f"- **{param}**: {config['description']} (Default: {config['default']})")
                    
                    if st.button(f"Explore {algo['name']}", key=f"explore_{algo['id']}"):
                        st.session_state.selected_algorithm = algo['id']
                        st.session_state.page = "Algorithm Explorer"
                        st.rerun()
        else:
            st.warning("No algorithms available. Please check if the backend server is running.")
    
    with col2:
        st.markdown("### Quick Start")
        st.info("""
        1. **Start the Backend**: Run `uvicorn main:app --reload` in the backend directory
        2. **Select Algorithm**: Choose from available algorithms
        3. **Configure Parameters**: Adjust hyperparameters
        4. **Train & Analyze**: See results and visualizations
        """)
        
        # Status check
        st.markdown("### Server Status")
        try:
            response = requests.get(f"{API_BASE_URL}/")
            if response.status_code == 200:
                st.success("✅ Backend API is running")
            else:
                st.error("❌ Backend API error")
        except:
            st.error("❌ Backend API not accessible")

def show_algorithm_explorer():
    """Modern algorithm explorer with enhanced UI"""
    st.markdown("""
    <div class="section-header">
        <h3>🔬 Algorithm Explorer</h3>
    </div>
    """, unsafe_allow_html=True)
    
    algorithms = get_algorithms_cached()
    if not algorithms:
        from performance_improvements import check_api_health
        if not check_api_health():
            st.error("❌ Backend API is not running. Please start the backend server first.")
            st.info("💡 **Instructions:**\n1. Navigate to the backend directory\n2. Run: `python main.py`\n3. Use the 'Refresh API Data' button in the sidebar when ready")
        else:
            st.error("❌ Unable to load algorithms. API is running but returned no data.")
            if st.button("🔄 Try Again"):
                from performance_improvements import clear_api_cache
                clear_api_cache()
                st.rerun()
        return
    
    # Algorithm selection with cards
    st.markdown("### 🎯 Select Algorithm")
    
    # Group algorithms by type for better organization
    algo_groups = {}
    for algo in algorithms:
        algo_type = algo['type']
        if algo_type not in algo_groups:
            algo_groups[algo_type] = []
        algo_groups[algo_type].append(algo)
    
    # Pre-select algorithm if coming from home page
    selected_algo = None
    if st.session_state.selected_algorithm:
        selected_algo = next((algo for algo in algorithms if algo['id'] == st.session_state.selected_algorithm), None)
    
    # Create tabs for algorithm types
    tabs = st.tabs([f"{key.title()} ({len(algos)})" for key, algos in algo_groups.items()])
    
    for i, (algo_type, algos) in enumerate(algo_groups.items()):
        with tabs[i]:
            cols = st.columns(min(3, len(algos)))
            for j, algo in enumerate(algos):
                with cols[j % 3]:
                    is_selected = selected_algo and selected_algo['id'] == algo['id']
                    if st.button(
                        f"Select {algo['name']}", 
                        key=f"select_{algo['id']}", 
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        st.session_state.selected_algorithm = algo['id']
                        selected_algo = algo
                        st.rerun()
                    
                    st.markdown(create_algorithm_card(algo, is_selected), unsafe_allow_html=True)
    
    # Show configuration and training interface if algorithm is selected
    if selected_algo:
        st.markdown("---")
        
        # Algorithm details
        st.markdown(f"""
        <div class="feature-highlight">
            <h4 style="margin-top: 0;">🎯 {selected_algo['name']}</h4>
            <p style="margin-bottom: 0;"><strong>Type:</strong> {selected_algo['type'].title()}</p>
            <p style="margin-bottom: 0;"><strong>Description:</strong> {selected_algo['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration and Training Interface
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div class="section-header">
                <h3>⚙️ Configuration</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Dataset configuration
            st.markdown("#### Dataset Settings")
            
            # Dataset source selection
            dataset_source = st.radio(
                "Dataset Source",
                ["Generated", "Built-in", "Upload Your Own"],
                help="Choose how to provide the dataset"
            )
            
            dataset_config = {}
            builtin_dataset = "diabetes"
            uploaded_data = None
            
            if dataset_source == "Generated":
                n_samples = st.slider("Number of Samples", 100, 2000, 500)
                n_features = st.slider("Number of Features", 2, 20, 10)
                noise = st.slider("Noise Level", 0.0, 2.0, 0.1)
                test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
                
                dataset_config = {
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "noise": noise,
                    "test_size": test_size,
                    "random_state": 42
                }
                dataset_source_api = "generated"
                
            elif dataset_source == "Built-in":
                datasets_info = get_datasets_cached()
                if datasets_info["builtin_datasets"]:
                    # Filter datasets by algorithm type (clustering datasets only for clustering algorithms)
                    available_datasets = datasets_info["builtin_datasets"]
                    if selected_algo['type'] == 'clustering':
                        # Show only clustering datasets for clustering algorithms
                        available_datasets = [d for d in available_datasets if d['type'] == 'clustering']
                    else:
                        # Show non-clustering datasets for non-clustering algorithms
                        available_datasets = [d for d in available_datasets if d['type'] != 'clustering']
                    
                    if available_datasets:
                        dataset_options = {d["id"]: f"{d['name']} ({d['type']})" for d in available_datasets}
                        builtin_dataset = st.selectbox(
                            "Select Dataset",
                            options=list(dataset_options.keys()),
                            format_func=lambda x: dataset_options[x]
                        )
                        
                        # Show dataset description
                        selected_ds = next(d for d in available_datasets if d["id"] == builtin_dataset)
                        st.info(f"📊 {selected_ds['description']}")
                        
                        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
                        dataset_config = {"test_size": test_size}
                        dataset_source_api = "builtin"
                    else:
                        st.warning(f"No built-in datasets available for {selected_algo['type']} algorithms. Please use generated data or upload your own.")
                        # Fall back to generated data
                        n_samples = st.slider("Number of Samples", 100, 2000, 500)
                        n_features = st.slider("Number of Features", 2, 20, 10)
                        noise = st.slider("Noise Level", 0.0, 2.0, 0.1)
                        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
                        
                        dataset_config = {
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "noise": noise,
                            "test_size": test_size,
                            "random_state": 42
                        }
                        dataset_source_api = "generated"
                
            else:  # Upload Your Own
                uploaded_file = st.file_uploader(
                    "Upload CSV file",
                    type=['csv'],
                    help="Upload a CSV file where the last column is the target variable"
                )
                
                if uploaded_file is not None:
                    upload_result = upload_dataset(uploaded_file)
                    if upload_result and upload_result.get("success"):
                        st.success(f"✅ Uploaded: {upload_result['filename']}")
                        st.info(f"Shape: {upload_result['shape']} | Columns: {', '.join(upload_result['columns'])}")
                        
                        # Show preview
                        if st.checkbox("Show Preview"):
                            preview_df = pd.DataFrame(upload_result['data'])
                            st.dataframe(preview_df)
                        
                        uploaded_data = upload_result['full_data']
                        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
                        dataset_config = {"test_size": test_size}
                        dataset_source_api = "uploaded"
                    else:
                        st.error("Failed to upload dataset")
                        dataset_source_api = "generated"
                        dataset_config = {"n_samples": 500, "n_features": 10, "noise": 0.1, "test_size": 0.2, "random_state": 42}
                else:
                    st.warning("Please upload a CSV file")
                    dataset_source_api = "generated"
                    dataset_config = {"n_samples": 500, "n_features": 10, "noise": 0.1, "test_size": 0.2, "random_state": 42}
            
            # Sklearn comparison option
            st.markdown("#### Comparison Settings")
            compare_sklearn = st.checkbox(
                "Compare with Scikit-learn",
                value=True,
                help="Compare your custom implementation with scikit-learn's version"
            )
            
            # Hyperparameter configuration
            st.markdown("#### Hyperparameters")
            hyperparameters = {}
            
            for param, config in selected_algo['hyperparameters'].items():
                if config['type'] == 'float':
                    value = st.slider(
                        f"{param.title()} - {config['description']}",
                        float(config['min']),
                        float(config['max']),
                        float(config['default']),
                        step=0.001
                    )
                elif config['type'] == 'int':
                    value = st.slider(
                        f"{param.title()} - {config['description']}",
                        int(config['min']),
                        int(config['max']),
                        int(config['default'])
                    )
                elif config['type'] == 'select':
                    value = st.selectbox(
                        f"{param.title()} - {config['description']}",
                        options=config['options'],
                        index=config['options'].index(config['default']) if config['default'] in config['options'] else 0
                    )
                hyperparameters[param] = value
            
            # Train button
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    result = train_model_optimized(
                        selected_algo['id'], 
                        hyperparameters, 
                        dataset_config,
                        compare_sklearn=compare_sklearn,
                        dataset_source=dataset_source_api,
                        builtin_dataset=builtin_dataset,
                        uploaded_data=uploaded_data
                    )
                    if result:
                        st.session_state.training_result = result
                        # Add to experiment history
                        add_experiment_to_history(
                            selected_algo['name'],
                            result,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        st.success("✅ Model trained successfully!")
                    else:
                        st.error("❌ Training failed")
        
        with col2:
            st.markdown("""
            <div class="section-header">
                <h3>📊 Results & Visualization</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display results if available
            if 'training_result' in st.session_state:
                result = st.session_state.training_result
                
                # Performance metrics with modern cards
                st.markdown("#### 🎯 Performance Metrics")
                
                # Custom metrics
                metrics = result.get('metrics', {})
                sklearn_metrics = result.get('sklearn_comparison', {}).get('metrics', {})
                
                # Display main metrics in cards
                metric_cols = st.columns(min(4, len(metrics)))
                for i, (metric, value) in enumerate(metrics.items()):
                    with metric_cols[i % 4]:
                        sklearn_value = sklearn_metrics.get(metric)
                        delta = (value - sklearn_value) if sklearn_value is not None else None
                        st.markdown(create_metric_card(
                            f"Custom {metric.upper()}", 
                            value, 
                            delta
                        ), unsafe_allow_html=True)
                
                # Sklearn comparison if available
                if sklearn_metrics:
                    st.markdown("#### 🔬 Scikit-learn Comparison")
                    sklearn_cols = st.columns(min(4, len(sklearn_metrics)))
                    for i, (metric, value) in enumerate(sklearn_metrics.items()):
                        with sklearn_cols[i % 4]:
                            st.markdown(create_metric_card(
                                f"Sklearn {metric.upper()}", 
                                value
                            ), unsafe_allow_html=True)
                
                # Model information in a nice format
                st.markdown("#### � Model Details")
                model_info = result.get('model_info', {})
                
                info_data = []
                for key, value in model_info.items():
                    if isinstance(value, (int, float)):
                        info_data.append({"Parameter": key.replace('_', ' ').title(), "Value": f"{value:.4f}" if isinstance(value, float) else str(value)})
                    else:
                        info_data.append({"Parameter": key.replace('_', ' ').title(), "Value": str(value)})
                
                if info_data:
                    info_df = pd.DataFrame(info_data)
                    st.dataframe(info_df, hide_index=True, use_container_width=True)
                
                # Visualizations based on algorithm type
                st.markdown("#### 📈 Visualizations")
                if selected_algo['type'] == 'regression':
                    show_regression_visualizations(result, dataset_config, selected_algo)
                elif selected_algo['type'] == 'classification':
                    show_classification_visualizations(result, dataset_config, selected_algo)
                elif selected_algo['type'] == 'clustering':
                    show_clustering_visualizations(result, dataset_config, selected_algo)
                
            else:
                st.info("👆 Configure your algorithm and click 'Train Model' to see results here!")
                
                # Show sample visualization or tips
                st.markdown("""
                <div class="feature-highlight">
                    <h4 style="margin-top: 0;">💡 What you'll see here:</h4>
                    <ul>
                        <li><strong>Performance Metrics:</strong> Compare your implementation with scikit-learn</li>
                        <li><strong>Interactive Visualizations:</strong> Plots and charts to understand algorithm behavior</li>
                        <li><strong>Model Details:</strong> Parameters, coefficients, and other model information</li>
                        <li><strong>Educational Insights:</strong> Learn how the algorithm works</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

def show_regression_visualizations(result, dataset_config, selected_algo):
    """Show visualizations for regression results"""
    st.markdown("#### 📈 Visualizations")
    
    # Training history (for algorithms that have it)
    if result.get('training_history'):
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=result['training_history'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='blue', width=2)
        ))
        
        loss_title = "Training Loss Over Time"
        y_title = "Mean Squared Error"
        if "linear" in selected_algo['id']:
            loss_title = "Linear Regression Training Loss"
        
        fig_loss.update_layout(
            title=loss_title,
            xaxis_title="Iteration",
            yaxis_title=y_title,
            template="plotly_white"
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Coefficients visualization (for linear models)
    if 'coefficients' in result['model_info']:
        coefficients = result['model_info']['coefficients']
        fig_coef = go.Figure()
        fig_coef.add_trace(go.Bar(
            x=[f"Bias"] + [f"Feature {i+1}" for i in range(len(coefficients)-1)],
            y=coefficients,
            marker_color='lightblue'
        ))
        fig_coef.update_layout(
            title="Model Coefficients",
            xaxis_title="Parameters",
            yaxis_title="Coefficient Value",
            template="plotly_white"
        )
        st.plotly_chart(fig_coef, use_container_width=True)
    
    # Feature importances (for tree models)
    if 'feature_importances' in result['model_info']:
        importances = result['model_info']['feature_importances']
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=[f"Feature {i+1}" for i in range(len(importances))],
            y=importances,
            marker_color='lightgreen'
        ))
        fig_imp.update_layout(
            title="Feature Importances",
            xaxis_title="Features",
            yaxis_title="Importance",
            template="plotly_white"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    
    # KNN-specific visualizations
    if 'knn' in selected_algo['id']:
        show_knn_visualizations(result, selected_algo, 'regression')

def show_classification_visualizations(result, dataset_config, selected_algo):
    """Show visualizations for classification results"""
    st.markdown("#### 📈 Visualizations")
    
    # Training history (for algorithms that have it)
    if result.get('training_history'):
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=result['training_history'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='red', width=2)
        ))
        
        loss_title = "Training Loss Over Time"
        y_title = "Cross-Entropy Loss"
        if "logistic" in selected_algo['id']:
            loss_title = "Logistic Regression Training Loss"
        
        fig_loss.update_layout(
            title=loss_title,
            xaxis_title="Iteration",
            yaxis_title=y_title,
            template="plotly_white"
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Coefficients visualization (for logistic regression)
    if 'coefficients' in result['model_info']:
        coefficients = result['model_info']['coefficients']
        fig_coef = go.Figure()
        fig_coef.add_trace(go.Bar(
            x=[f"Bias"] + [f"Feature {i+1}" for i in range(len(coefficients)-1)],
            y=coefficients,
            marker_color='lightcoral'
        ))
        fig_coef.update_layout(
            title="Model Coefficients",
            xaxis_title="Parameters",
            yaxis_title="Coefficient Value",
            template="plotly_white"
        )
        st.plotly_chart(fig_coef, use_container_width=True)
    
    # Feature importances (for tree models)
    if 'feature_importances' in result['model_info']:
        importances = result['model_info']['feature_importances']
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=[f"Feature {i+1}" for i in range(len(importances))],
            y=importances,
            marker_color='lightgreen'
        ))
        fig_imp.update_layout(
            title="Feature Importances",
            xaxis_title="Features",
            yaxis_title="Importance",
            template="plotly_white"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    
    # Prediction probabilities (for probabilistic models)
    if 'probabilities' in result['model_info']:
        probabilities = result['model_info']['probabilities']
        if probabilities:
            proba_array = np.array(probabilities)
            if proba_array.shape[1] == 2:  # Binary classification
                fig_proba = go.Figure()
                fig_proba.add_trace(go.Histogram(
                    x=proba_array[:, 1],
                    nbinsx=20,
                    name='Class 1 Probability',
                    marker_color='orange'
                ))
                fig_proba.update_layout(
                    title="Prediction Probability Distribution",
                    xaxis_title="Probability of Class 1",
                    yaxis_title="Count",
                    template="plotly_white"
                )
                st.plotly_chart(fig_proba, use_container_width=True)
    
    # KNN-specific visualizations
    if 'knn' in selected_algo['id']:
        show_knn_visualizations(result, selected_algo, 'classification')

def show_clustering_visualizations(result, dataset_config, selected_algo):
    """Show visualizations for clustering results"""
    st.markdown("#### 📈 Visualizations")
    
    # Inertia history
    if result.get('training_history'):
        fig_inertia = go.Figure()
        fig_inertia.add_trace(go.Scatter(
            y=result['training_history'],
            mode='lines+markers',
            name='Inertia',
            line=dict(color='green', width=2)
        ))
        fig_inertia.update_layout(
            title="k-Means Convergence",
            xaxis_title="Iteration",
            yaxis_title="Within-Cluster Sum of Squares (Inertia)",
            template="plotly_white"
        )
        st.plotly_chart(fig_inertia, use_container_width=True)
    
    # Centroids visualization
    if 'centroids' in result['model_info']:
        centroids = result['model_info']['centroids']
        centroids_array = np.array(centroids)
        
        if centroids_array.shape[1] >= 2:
            fig_centroids = go.Figure()
            
            # Plot centroids
            fig_centroids.add_trace(go.Scatter(
                x=centroids_array[:, 0],
                y=centroids_array[:, 1],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='x',
                    line=dict(width=2, color='black')
                ),
                name='Centroids'
            ))
            
            fig_centroids.update_layout(
                title="Cluster Centroids",
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                template="plotly_white"
            )
            st.plotly_chart(fig_centroids, use_container_width=True)
        
        # Centroids table
        st.markdown("#### 📊 Centroids Coordinates")
        centroids_df = pd.DataFrame(
            centroids,
            columns=[f"Feature {i+1}" for i in range(len(centroids[0]))],
            index=[f"Cluster {i}" for i in range(len(centroids))]
        )
        st.dataframe(centroids_df)

def show_knn_visualizations(result, selected_algo, task_type):
    """Show visualizations specific to KNN algorithms"""
    st.markdown("#### 🎯 KNN Algorithm Analysis")
    
    model_info = result['model_info']
    k_value = model_info.get('k', 5)
    distance_metric = model_info.get('distance_metric', 'euclidean')
    
    # KNN Parameters Display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("K Value", k_value)
    with col2:
        st.metric("Distance Metric", distance_metric.title())
    with col3:
        st.metric("Training Samples", model_info.get('training_samples', 'N/A'))
    
    # Performance comparison with sklearn
    if result.get('sklearn_comparison'):
        st.markdown("#### 📊 Performance Comparison")
        
        # Create comparison chart
        metrics = list(result['metrics'].keys())
        custom_values = [result['metrics'][m] for m in metrics]
        sklearn_values = [result['sklearn_comparison'].get(m, 0) for m in metrics]
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Custom Implementation',
            x=metrics,
            y=custom_values,
            marker_color='lightblue'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Scikit-learn',
            x=metrics,
            y=sklearn_values,
            marker_color='lightcoral'
        ))
        
        fig_comparison.update_layout(
            title=f"KNN {task_type.title()} Performance Comparison",
            xaxis_title="Metrics",
            yaxis_title="Score",
            barmode='group',
            template="plotly_white"
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # KNN Algorithm Explanation
    st.markdown("#### 🔍 How KNN Works")
    
    if task_type == 'classification':
        st.markdown(f"""
        **k-Nearest Neighbors Classification** with k={k_value}:
        
        1. **Distance Calculation**: For each test point, calculate {distance_metric} distance to all training points
        2. **Find Neighbors**: Select the {k_value} closest training points
        3. **Majority Vote**: Assign the most common class among the {k_value} neighbors
        4. **Prediction**: Output the predicted class
        
        **Key Characteristics**:
        - Lazy learning (no explicit training phase)
        - Non-parametric (makes no assumptions about data distribution)
        - Local decision boundaries
        - Sensitive to curse of dimensionality
        """)
    else:
        st.markdown(f"""
        **k-Nearest Neighbors Regression** with k={k_value}:
        
        1. **Distance Calculation**: For each test point, calculate {distance_metric} distance to all training points
        2. **Find Neighbors**: Select the {k_value} closest training points
        3. **Average Values**: Take the mean of the {k_value} neighbor target values
        4. **Prediction**: Output the averaged value
        
        **Key Characteristics**:
        - Lazy learning (no explicit training phase)
        - Non-parametric (makes no assumptions about data distribution)
        - Smooth decision surfaces
        - Good for local patterns in data
        """)
    
    # K-Value Effect Analysis
    st.markdown("#### 📈 Effect of K Value")
    st.markdown(f"""
    **Current K = {k_value}**:
    
    - **Low K (1-3)**: More sensitive to noise, complex decision boundaries, higher variance
    - **Medium K (5-10)**: Balanced bias-variance tradeoff, good generalization
    - **High K (>10)**: Smoother boundaries, less sensitive to noise, higher bias
    
    **Recommendation**: Try different K values to find the optimal balance for your dataset.
    """)
    
    # Distance Metric Explanation
    st.markdown("#### 📏 Distance Metrics")
    
    distance_explanations = {
        'euclidean': "**Euclidean Distance**: Standard straight-line distance. Good for continuous features with similar scales.",
        'manhattan': "**Manhattan Distance**: Sum of absolute differences. More robust to outliers than Euclidean.",
        'minkowski': "**Minkowski Distance**: Generalization of Euclidean and Manhattan. Flexible parameter p."
    }
    
    st.markdown(f"**Current: {distance_metric.title()} Distance**")
    st.markdown(distance_explanations.get(distance_metric, f"Using {distance_metric} distance metric."))

def show_about_page():
    """Display the about page"""
    st.header("📚 About ML Playground")
    
    st.markdown("""
    ### What is ML Playground?
    
    ML Playground is an educational platform designed to help you understand machine learning algorithms 
    by implementing them from scratch. Instead of using black-box implementations, you can see exactly 
    how each algorithm works under the hood.
    
    ### Features
    
    - **📝 From-Scratch Implementations**: All algorithms are implemented from first principles
    - **🎛️ Interactive Controls**: Adjust hyperparameters and see immediate effects
    - **📊 Rich Visualizations**: Understand training dynamics and model behavior
    - **⚡ Real-time Training**: Train models instantly with different configurations
    - **🔬 Educational Focus**: Learn how algorithms actually work
    
    ### Architecture
    
    - **Backend**: FastAPI serving custom ML implementations
    - **Frontend**: Streamlit for interactive web interface
    - **Algorithms**: Pure NumPy implementations for transparency
    
    ### Current Algorithms
    
    #### Regression
    1. **Linear Regression** - Gradient descent implementation with customizable learning rate and iterations
    2. **k-NN Regression** - k-Nearest Neighbors for regression with distance weighting
    3. **Decision Tree Regression** - Tree-based regression with MSE splitting criterion
    
    #### Classification  
    1. **Logistic Regression** - Sigmoid-based binary classification with gradient descent
    2. **k-NN Classification** - k-Nearest Neighbors for classification with majority voting
    3. **Decision Tree Classification** - Tree-based classification with entropy/Gini splitting
    
    #### Clustering
    1. **k-Means Clustering** - Centroid-based clustering with Lloyd's algorithm
    
    ### Coming Soon
    
    - Naive Bayes
    - Support Vector Machines
    - Random Forest
    - Neural Networks
    - PCA (Principal Component Analysis)
    
    ### How to Use
    
    1. **Start the Backend**: Run the FastAPI server
    2. **Select Algorithm**: Choose from the available algorithms
    3. **Configure Parameters**: Adjust dataset and hyperparameters
    4. **Train & Explore**: Train the model and analyze results
    
    ### Technologies Used
    
    - **Python**: Core language
    - **NumPy**: Mathematical operations
    - **FastAPI**: REST API backend
    - **Streamlit**: Web interface
    - **Plotly**: Interactive visualizations
    - **scikit-learn**: Dataset generation and metrics
    
    ### Contributing
    
    This project is open for contributions! You can:
    - Add new algorithms
    - Improve visualizations
    - Enhance the user interface
    - Add more educational content
    
    ---
    
    Made with ❤️ for learning and understanding ML algorithms
    """)

if __name__ == "__main__":
    main()
