# Algorithm comparison page implementation
import streamlit as st
from ui_components import render_page_header
from performance_improvements import get_algorithms_cached, get_datasets_cached, train_model_optimized

def render_compare_page():
    """Render the algorithm comparison page"""
    render_page_header(
        "Algorithm Comparison Lab", 
        "Compare multiple algorithms side-by-side on the same dataset",
        "🔬"
    )
    
    algorithms = get_algorithms_cached()
    if not algorithms:
        st.error("Unable to load algorithms. Please check the backend connection.")
        return
    
    # Algorithm selection for comparison
    st.markdown("### 🎯 Select Algorithms to Compare")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Algorithm 1**")
        algo1 = st.selectbox(
            "First Algorithm",
            algorithms,
            format_func=lambda x: f"{x['name']} ({x['type'].title()})",
            key="compare_algo1"
        )
    
    with col2:
        st.markdown("**Algorithm 2**")
        # Filter algorithms by same type as algo1
        compatible_algos = [a for a in algorithms if a['type'] == algo1['type']]
        algo2 = st.selectbox(
            "Second Algorithm",
            compatible_algos,
            format_func=lambda x: f"{x['name']} ({x['type'].title()})",
            key="compare_algo2"
        )
    
    if algo1['id'] == algo2['id']:
        st.warning("⚠️ Please select different algorithms for comparison.")
        return
    
    st.markdown("### 📊 Dataset Configuration")
    
    # Dataset selection
    dataset_source = st.radio(
        "Choose Dataset Source",
        ["Generated", "Built-in"],
        key="compare_dataset_source"
    )
    
    if dataset_source == "Generated":
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Number of Samples", 50, 1000, 300, key="compare_samples")
            n_features = st.slider("Number of Features", 2, 20, 4, key="compare_features")
        
        with col2:
            noise = st.slider("Noise Level", 0.0, 1.0, 0.1, 0.05, key="compare_noise")
            random_state = st.slider("Random State", 0, 100, 42, key="compare_random")
        
        dataset_config = {
            "n_samples": n_samples,
            "n_features": n_features,
            "noise": noise,
            "random_state": random_state
        }
        dataset_source_api = "generated"
        builtin_dataset = None
    else:
        datasets_info = get_datasets_cached()
        available_datasets = [d for d in datasets_info["builtin_datasets"] if d['type'] == algo1['type']]
        
        if available_datasets:
            selected_dataset = st.selectbox("Select Dataset", available_datasets, format_func=lambda x: x['name'], key="comp_builtin")
            builtin_dataset = selected_dataset['id']
            dataset_config = {}
            dataset_source_api = "builtin"
            st.info(f"**Description:** {selected_dataset['description']}")
        else:
            st.warning(f"No built-in datasets available for {algo1['type']} algorithms.")
            st.info("Switching to generated dataset.")
            dataset_config = {
                "n_samples": 300,
                "n_features": 4,
                "noise": 0.1,
                "random_state": 42
            }
            dataset_source_api = "generated"
            builtin_dataset = None
    
    st.markdown("---")
    
    # Training options
    st.markdown("### ⚙️ Training Options")
    compare_sklearn = st.checkbox("Compare with scikit-learn", value=True, key="compare_sklearn_option")
    
    # Train button
    if st.button("🚀 Compare Algorithms", type="primary", use_container_width=True):
        col1, col2 = st.columns(2)
        
        results = {}
        
        # Train both algorithms
        for i, (algo, col) in enumerate([(algo1, col1), (algo2, col2)], 1):
            with col:
                st.markdown(f"### 🔧 {algo['name']}")
                
                with st.spinner(f"Training {algo['name']}..."):
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
                        results[algo['name']] = result
                        
                        # Display metrics
                        st.success("✅ Training completed!")
                        st.metric("Custom Score", f"{result.get('custom_score', 0):.4f}")
                        
                        if 'sklearn_score' in result:
                            st.metric("Scikit-learn Score", f"{result['sklearn_score']:.4f}")
                        
                        if 'training_time' in result:
                            st.metric("Training Time", f"{result['training_time']:.3f}s")
                        
                        # Show hyperparameters used
                        with st.expander("View Hyperparameters"):
                            for param, value in hyperparameters.items():
                                st.write(f"**{param}:** {value}")
                    
                    else:
                        st.error(f"❌ Failed to train {algo['name']}")
        
        # Comparison summary
        if len(results) == 2:
            st.markdown("---")
            st.markdown("### 📊 Comparison Summary")
            
            algo_names = list(results.keys())
            scores = [results[name].get('custom_score', 0) for name in algo_names]
            sklearn_scores = [results[name].get('sklearn_score', 0) for name in algo_names if 'sklearn_score' in results[name]]
            times = [results[name].get('training_time', 0) for name in algo_names]
            
            # Performance comparison chart
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Model Performance', 'Training Time'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Performance bars
            fig.add_trace(
                go.Bar(x=algo_names, y=scores, name="Custom Implementation", marker_color='#667eea'),
                row=1, col=1
            )
            
            if sklearn_scores and len(sklearn_scores) == 2:
                fig.add_trace(
                    go.Bar(x=algo_names, y=sklearn_scores, name="Scikit-learn", marker_color='#764ba2'),
                    row=1, col=1
                )
            
            # Training time bars
            fig.add_trace(
                go.Bar(x=algo_names, y=times, name="Training Time", marker_color='#28a745', showlegend=False),
                row=1, col=2
            )
            
            fig.update_layout(height=400, title_text="Algorithm Comparison Results")
            fig.update_xaxes(title_text="Algorithm", row=1, col=1)
            fig.update_xaxes(title_text="Algorithm", row=1, col=2)
            fig.update_yaxes(title_text="Score", row=1, col=1)
            fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Winner announcement
            best_algo = algo_names[scores.index(max(scores))]
            fastest_algo = algo_names[times.index(min(times))]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🏆 **Best Performance:** {best_algo} ({max(scores):.4f})")
            
            with col2:
                st.info(f"⚡ **Fastest Training:** {fastest_algo} ({min(times):.3f}s)")
            
            # Detailed comparison table
            with st.expander("📋 Detailed Results"):
                import pandas as pd
                
                comparison_data = []
                for name in algo_names:
                    result = results[name]
                    comparison_data.append({
                        'Algorithm': name,
                        'Custom Score': f"{result.get('custom_score', 0):.4f}",
                        'Scikit-learn Score': f"{result.get('sklearn_score', 0):.4f}" if 'sklearn_score' in result else "N/A",
                        'Training Time': f"{result.get('training_time', 0):.3f}s",
                        'Iterations': result.get('iterations', 'N/A')
                    })
                
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
