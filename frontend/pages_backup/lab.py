# Lab/Experiment page implementation
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from ui_components import render_page_header
from performance_improvements import get_algorithms_cached, get_datasets_cached, train_model_optimized
from api_utils import add_experiment_to_history

def render_hyperparameter_controls(algorithm):
    """Render hyperparameter controls for an algorithm"""
    hyperparameters = {}
    
    st.markdown("### ⚙️ Hyperparameters")
    
    for param, config in algorithm['hyperparameters'].items():
        if config['type'] == 'int':
            hyperparameters[param] = st.slider(
                f"{param} ({config.get('description', '')})",
                min_value=config.get('min', 1),
                max_value=config.get('max', 100),
                value=config.get('default', 10),
                key=f"param_{param}"
            )
        elif config['type'] == 'float':
            hyperparameters[param] = st.slider(
                f"{param} ({config.get('description', '')})",
                min_value=config.get('min', 0.01),
                max_value=config.get('max', 1.0),
                value=config.get('default', 0.1),
                step=0.01,
                key=f"param_{param}"
            )
        elif config['type'] == 'categorical':
            hyperparameters[param] = st.selectbox(
                f"{param} ({config.get('description', '')})",
                options=config.get('options', []),
                index=0,
                key=f"param_{param}"
            )
    
    return hyperparameters

def render_dataset_configuration(algorithm_type):
    """Render dataset configuration options"""
    st.markdown("### 📊 Dataset Configuration")
    
    dataset_source = st.radio(
        "Choose Dataset Source",
        ["Generated", "Built-in", "Upload"],
        key="dataset_source_lab"
    )
    
    dataset_config = {}
    dataset_source_api = "generated"
    builtin_dataset = None
    uploaded_data = None
    
    if dataset_source == "Generated":
        st.markdown("**Synthetic Dataset Parameters**")
        
        col1, col2 = st.columns(2)
        with col1:
            dataset_config["n_samples"] = st.slider("Number of Samples", 50, 1000, 200)
            dataset_config["n_features"] = st.slider("Number of Features", 2, 20, 4)
        
        with col2:
            dataset_config["noise"] = st.slider("Noise Level", 0.0, 1.0, 0.1, 0.05)
            dataset_config["random_state"] = st.slider("Random State", 0, 100, 42)
    
    elif dataset_source == "Built-in":
        datasets_info = get_datasets_cached()
        if datasets_info["builtin_datasets"]:
            available_datasets = datasets_info["builtin_datasets"]
            if algorithm_type == 'clustering':
                available_datasets = [d for d in available_datasets if 'clustering' in d.get('type', '')]
            
            if available_datasets:
                selected_dataset = st.selectbox(
                    "Select Dataset", 
                    available_datasets, 
                    format_func=lambda x: x['name'],
                    key="lab_builtin_dataset"
                )
                builtin_dataset = selected_dataset['id']
                dataset_source_api = "builtin"
                st.info(f"**Description:** {selected_dataset['description']}")
            else:
                st.warning(f"No built-in datasets available for {algorithm_type}")
                dataset_source_api = "generated"
                dataset_config = {"n_samples": 200, "n_features": 4, "noise": 0.1, "random_state": 42}
        else:
            st.error("Could not load built-in datasets")
            dataset_source_api = "generated"
            dataset_config = {"n_samples": 200, "n_features": 4, "noise": 0.1, "random_state": 42}
    
    elif dataset_source == "Upload":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
                uploaded_data = df.to_dict('records')
                dataset_source_api = "uploaded"
            except Exception as e:
                st.error(f"Error loading file: {e}")
                dataset_source_api = "generated"
                dataset_config = {"n_samples": 200, "n_features": 4, "noise": 0.1, "random_state": 42}
        else:
            st.info("Please upload a CSV file")
            dataset_source_api = "generated"
            dataset_config = {"n_samples": 200, "n_features": 4, "noise": 0.1, "random_state": 42}
    
    return dataset_config, dataset_source_api, builtin_dataset, uploaded_data

def render_lab_page():
    """Render the main experiment/lab page"""
    render_page_header(
        "Algorithm Laboratory", 
        "Experiment with machine learning algorithms and track your results",
        "🧪"
    )
    
    algorithms = get_algorithms_cached()
    if not algorithms:
        st.error("❌ Unable to load algorithms. Please check the backend connection.")
        return
    
    # Algorithm selection
    st.markdown("### 🎯 Select Algorithm")
    
    # Check if algorithm was pre-selected from home page
    if st.session_state.selected_algorithm:
        default_algo = st.session_state.selected_algorithm
        default_index = next((i for i, algo in enumerate(algorithms) if algo['id'] == default_algo['id']), 0)
        st.session_state.selected_algorithm = None  # Clear after use
    else:
        default_index = 0
    
    selected_algo = st.selectbox(
        "Choose an algorithm to experiment with:",
        algorithms,
        index=default_index,
        format_func=lambda x: f"{x['name']} ({x['type'].title()})",
        key="lab_algorithm_select"
    )
    
    if selected_algo:
        # Algorithm info
        st.markdown(f"""
        <div class="algorithm-card">
            <h4>🔧 {selected_algo['name']}</h4>
            <p><strong>Type:</strong> {selected_algo['type'].title()}</p>
            <p><strong>Description:</strong> {selected_algo['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Hyperparameters
            hyperparameters = render_hyperparameter_controls(selected_algo)
        
        with col2:
            # Dataset configuration
            dataset_config, dataset_source_api, builtin_dataset, uploaded_data = render_dataset_configuration(selected_algo['type'])
            
            # Training options
            st.markdown("### 🎛️ Training Options")
            compare_sklearn = st.checkbox("Compare with scikit-learn", value=True, key="lab_compare_sklearn")
        
        st.markdown("---")
        
        # Train button
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
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
                timestamp = datetime.now().isoformat()
                add_experiment_to_history(selected_algo['name'], result, timestamp)
                
                # Display results
                st.success("🎉 Training completed successfully!")
                
                # Show metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Custom Score", f"{result.get('custom_score', 0):.4f}")
                
                with col2:
                    if compare_sklearn and 'sklearn_score' in result:
                        st.metric("Scikit-learn Score", f"{result['sklearn_score']:.4f}")
                
                with col3:
                    if 'training_time' in result:
                        st.metric("Training Time", f"{result['training_time']:.3f}s")
                
                # Additional visualizations based on algorithm type
                if 'visualizations' in result:
                    st.markdown("### 📊 Results Visualization")
                    
                    viz_data = result['visualizations']
                    
                    # Loss/convergence plot
                    if 'loss_history' in viz_data:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=viz_data['loss_history'],
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='#667eea', width=2)
                        ))
                        fig.update_layout(
                            title="Training Loss Over Iterations",
                            xaxis_title="Iteration",
                            yaxis_title="Loss",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.info("✅ Experiment saved to history. View all results in the Dashboard.")
            
            else:
                st.error("❌ Training failed. Please check your parameters and try again.")
    
    # Recent experiments sidebar
    with st.sidebar:
        st.markdown("### 📈 Recent Experiments")
        
        recent_experiments = st.session_state.experiment_history[:5]
        if recent_experiments:
            for i, exp in enumerate(recent_experiments):
                status_icon = "✅" if exp['status'] == 'Success' else "❌"
                st.markdown(f"""
                <div class="experiment-item">
                    <small>{status_icon} {exp['algorithm']}</small><br>
                    <small>Score: {exp['accuracy']:.4f}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No experiments yet")
        
        if st.button("📊 View All Results", use_container_width=True):
            st.session_state.current_page = "Dashboard"
            st.rerun()
