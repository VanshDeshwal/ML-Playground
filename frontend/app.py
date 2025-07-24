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

# Configure page
st.set_page_config(
    page_title="ML Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    """Load custom CSS for modern design"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .algorithm-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        border: 1px solid #f0f0f0;
    }
    
    .algorithm-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    .algorithm-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    
    .comparison-better {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-color: #c3e6cb;
        color: #155724;
        font-weight: 600;
    }
    
    .comparison-worse {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-color: #f5c6cb;
        color: #721c24;
        font-weight: 600;
    }
    
    .section-header {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1.5rem 0 1rem 0;
    }
    
    .section-header h3 {
        margin: 0;
        color: #495057;
        font-weight: 600;
    }
    
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
    }
    
    .nav-pill {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 25px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        display: inline-block;
        transition: all 0.3s ease;
    }
    
    .nav-pill:hover {
        background: #667eea;
        color: white;
        transform: translateY(-1px);
    }
    
    .experiment-row {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .experiment-row:hover {
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border-color: #667eea;
    }
    
    .progress-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
    }
    
    .progress-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        transition: width 0.3s ease;
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
    }
    
    .feature-highlight::before {
        content: "✨";
        position: absolute;
        top: -10px;
        left: 20px;
        background: white;
        padding: 0 10px;
        font-size: 1.2rem;
    }
    
    .sidebar .sidebar-content {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Custom button styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Custom selectbox styles */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e9ecef;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: 1px solid #dee2e6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def get_algorithms():
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
        algorithms = get_algorithms()
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
    """Display the algorithm explorer page"""
    st.header("🔬 Algorithm Explorer")
    
    # Debug info (can be removed later)
    if 'selected_algorithm' in st.session_state:
        st.info(f"🎯 Pre-selected algorithm: {st.session_state.selected_algorithm}")
    
    # Get selected algorithm from session state or selectbox
    algorithms = get_algorithms()
    if not algorithms:
        st.error("No algorithms available")
        return
    
    algorithm_names = {algo['id']: algo['name'] for algo in algorithms}
    
    # Check if we have a pre-selected algorithm from session state
    default_index = 0
    if 'selected_algorithm' in st.session_state and st.session_state.selected_algorithm in algorithm_names:
        default_index = list(algorithm_names.keys()).index(st.session_state.selected_algorithm)
    
    # Algorithm selection
    selected_id = st.selectbox(
        "Choose an Algorithm",
        options=list(algorithm_names.keys()),
        format_func=lambda x: algorithm_names[x],
        index=default_index,
        key="algorithm_selector"
    )
    
    # Update session state with current selection
    st.session_state.selected_algorithm = selected_id
    
    # Find selected algorithm
    selected_algo = next((algo for algo in algorithms if algo['id'] == selected_id), None)
    
    if selected_algo:
        st.subheader(f"🎯 {selected_algo['name']}")
        st.write(selected_algo['description'])
        
        # Create columns for configuration and results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Configuration")
            
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
                datasets_info = get_datasets()
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
                    result = train_model(
                        selected_id, 
                        hyperparameters, 
                        dataset_config,
                        compare_sklearn=compare_sklearn,
                        dataset_source=dataset_source_api,
                        builtin_dataset=builtin_dataset,
                        uploaded_data=uploaded_data
                    )
                    if result:
                        st.session_state.training_result = result
                        st.success("Model trained successfully!")
                    else:
                        st.error("Training failed")
        
        with col2:
            st.markdown("### Results & Visualization")
            
            # Display results if available
            if 'training_result' in st.session_state:
                result = st.session_state.training_result
                
                # Metrics comparison
                st.markdown("#### 📊 Performance Metrics")
                
                if result.get('sklearn_comparison'):
                    # Show comparison side by side
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.markdown("**Custom Implementation**")
                        for metric, value in result['metrics'].items():
                            st.metric(metric.upper(), f"{value:.4f}")
                    
                    with comp_col2:
                        st.markdown("**Scikit-learn Implementation**")
                        for metric, value in result['sklearn_comparison'].items():
                            st.metric(metric.upper(), f"{value:.4f}")
                    
                    # Comparison summary
                    st.markdown("#### 🔍 Comparison Summary")
                    for metric in result['metrics'].keys():
                        custom_val = result['metrics'][metric]
                        
                        # Check if this metric exists in sklearn comparison
                        if metric in result['sklearn_comparison']:
                            sklearn_val = result['sklearn_comparison'][metric]
                            diff = abs(custom_val - sklearn_val)
                            
                            if metric in ['mse', 'rmse', 'mae', 'inertia']:  # Lower is better
                                if custom_val < sklearn_val:
                                    status = "🟢 Custom performs better"
                                elif custom_val > sklearn_val:
                                    status = "🔴 Sklearn performs better"
                                else:
                                    status = "🟡 Equal performance"
                            else:  # Higher is better (accuracy, r2, silhouette_score, etc.)
                                if custom_val > sklearn_val:
                                    status = "🟢 Custom performs better"
                                elif custom_val < sklearn_val:
                                    status = "🔴 Sklearn performs better"
                                else:
                                    status = "🟡 Equal performance"
                            
                            st.write(f"**{metric.upper()}:** {status} (diff: {diff:.4f})")
                        else:
                            # Metric not available in sklearn comparison
                            st.write(f"**{metric.upper()}:** Custom only: {custom_val:.4f}")
                        
                else:
                    # Show only custom metrics
                    metrics_cols = st.columns(len(result['metrics']))
                    for i, (metric, value) in enumerate(result['metrics'].items()):
                        with metrics_cols[i]:
                            st.metric(metric.upper(), f"{value:.4f}")
                
                # Model info
                st.markdown("#### 🔧 Model Information")
                model_info = result['model_info']
                if 'n_features' in model_info:
                    st.write(f"**Number of Features:** {model_info['n_features']}")
                if 'training_samples' in model_info:
                    st.write(f"**Training Samples:** {model_info['training_samples']}")
                
                # Show algorithm-specific info
                for key, value in model_info.items():
                    if key not in ['n_features', 'training_samples']:
                        if isinstance(value, (int, float)):
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        elif isinstance(value, str):
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        # Skip complex objects like arrays
                
                # Visualizations
                if selected_algo['type'] == 'regression':
                    show_regression_visualizations(result, dataset_config, selected_algo)
                elif selected_algo['type'] == 'classification':
                    show_classification_visualizations(result, dataset_config, selected_algo)
                elif selected_algo['type'] == 'clustering':
                    show_clustering_visualizations(result, dataset_config, selected_algo)
                
            else:
                st.info("Train a model to see results and visualizations")

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
