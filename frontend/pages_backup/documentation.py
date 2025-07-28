# Documentation page implementation
import streamlit as st
from ui_components import render_page_header
from performance_improvements import get_algorithms_cached, get_datasets_cached

def render_documentation_page():
    """Render the documentation and learning resources page"""
    render_page_header(
        "Documentation & Learning Resources", 
        "Learn about machine learning algorithms and best practices",
        "📚"
    )
    
    # Algorithm documentation tabs
    algorithms = get_algorithms_cached()
    if algorithms:
        tabs = st.tabs(["📖 Overview", "🧮 Algorithms", "📊 Datasets", "🎯 Tips & Best Practices"])
        
        with tabs[0]:
            st.markdown("""
            ## Welcome to ML Playground! 🤖
            
            This interactive platform helps you understand machine learning algorithms through hands-on experimentation.
            
            ### 🎯 What You Can Do
            
            - **Learn Algorithm Internals**: See how algorithms work with from-scratch implementations
            - **Compare Performance**: Test different algorithms on the same dataset
            - **Experiment Safely**: Try different hyperparameters without consequences
            - **Track Progress**: Monitor your experiments and learning journey
            
            ### 🚀 Getting Started
            
            1. **Explore**: Start with the Home page to see available algorithms
            2. **Experiment**: Go to the Lab to train your first model
            3. **Compare**: Use the Compare page to understand algorithm differences
            4. **Analyze**: Check the Dashboard to track your progress
            
            ### 🔧 Algorithm Types
            
            - **Regression**: Predict continuous values (prices, temperatures, etc.)
            - **Classification**: Predict categories (spam/not spam, disease diagnosis)
            - **Clustering**: Group similar data points (customer segmentation)
            
            ### 📚 Learning Path
            
            1. Start with **Linear Regression** for basic concepts
            2. Try **Logistic Regression** for classification
            3. Explore **K-Means** for clustering
            4. Advanced: **K-Nearest Neighbors** and **Decision Trees**
            """)
        
        with tabs[1]:
            st.markdown("### 🧮 Algorithm Deep Dive")
            
            for algo in algorithms:
                with st.expander(f"🔧 {algo['name']} ({algo['type'].title()})"):
                    st.markdown(f"**Description:** {algo['description']}")
                    st.markdown(f"**Type:** {algo['type'].title()}")
                    
                    if algo['hyperparameters']:
                        st.markdown("**Key Hyperparameters:**")
                        for param, config in algo['hyperparameters'].items():
                            st.markdown(f"- **{param}**: {config.get('description', 'No description available')}")
                    
                    # Add algorithm-specific learning content
                    if algo['id'] == 'linear_regression':
                        st.markdown("""
                        **When to Use:**
                        - Predicting continuous values
                        - Understanding feature relationships
                        - Baseline for other algorithms
                        
                        **Key Concepts:**
                        - Fits a line through data points
                        - Minimizes squared errors
                        - Assumes linear relationship
                        """)
                    
                    elif algo['id'] == 'logistic_regression':
                        st.markdown("""
                        **When to Use:**
                        - Binary classification problems
                        - Probability estimates needed
                        - Interpretable results required
                        
                        **Key Concepts:**
                        - Uses sigmoid function
                        - Outputs probabilities
                        - Linear decision boundary
                        """)
                    
                    elif algo['id'] == 'kmeans':
                        st.markdown("""
                        **When to Use:**
                        - Customer segmentation
                        - Data exploration
                        - Dimensionality reduction
                        
                        **Key Concepts:**
                        - Partitions data into k clusters
                        - Minimizes within-cluster variance
                        - Requires number of clusters
                        """)
                    
                    elif algo['id'] == 'knn':
                        st.markdown("""
                        **When to Use:**
                        - Non-linear patterns
                        - Local decision boundaries
                        - Simple baseline method
                        
                        **Key Concepts:**
                        - "Lazy" learning algorithm
                        - Uses distance to neighbors
                        - Sensitive to feature scaling
                        """)
                    
                    elif algo['id'] == 'decision_tree':
                        st.markdown("""
                        **When to Use:**
                        - Interpretable decisions needed
                        - Mixed data types
                        - Feature interactions important
                        
                        **Key Concepts:**
                        - Tree-like decision structure
                        - Splits based on features
                        - Can overfit easily
                        """)
        
        with tabs[2]:
            st.markdown("### 📊 Available Datasets")
            datasets_info = get_datasets_cached()
            
            if datasets_info["builtin_datasets"]:
                for dataset in datasets_info["builtin_datasets"]:
                    with st.expander(f"{dataset['name']} ({dataset['type']})"):
                        st.markdown(f"**Description**: {dataset['description']}")
                        st.markdown(f"**Type**: {dataset['type'].title()}")
                        st.markdown(f"**Best for**: {dataset.get('use_case', 'General experimentation')}")
            
            st.markdown("""
            ### 🔄 Dataset Options
            
            **Generated Datasets:**
            - Synthetic data created on-the-fly
            - Perfect for learning and experimentation
            - Controllable parameters (samples, features, noise)
            
            **Built-in Datasets:**
            - Real-world datasets from scikit-learn
            - Good for testing algorithm performance
            - Various difficulty levels
            
            **Upload Your Own:**
            - Use your own CSV files
            - Great for real projects
            - Make sure data is clean and formatted
            """)
        
        with tabs[3]:
            st.markdown("""
            ### 🎯 Tips & Best Practices
            
            #### 🔧 Hyperparameter Tuning
            - Start with default values
            - Change one parameter at a time
            - Use validation data for evaluation
            - Document what works and what doesn't
            
            #### 📊 Data Preparation
            - **Clean your data**: Remove or handle missing values
            - **Scale features**: Especially important for KNN and clustering
            - **Understand your data**: Plot distributions and correlations
            - **Split properly**: Train/validation/test sets
            
            #### 🎪 Experimentation Strategy
            - **Start simple**: Begin with basic algorithms
            - **Compare fairly**: Use same dataset and evaluation
            - **Track everything**: Record parameters and results
            - **Understand failures**: Learn from poor results
            
            #### 🏆 Performance Evaluation
            - **Don't just look at accuracy**: Consider precision, recall, F1
            - **Use appropriate metrics**: MSE for regression, accuracy for classification
            - **Consider computational cost**: Training time vs. performance
            - **Test on unseen data**: Avoid overfitting
            
            #### 📈 Algorithm Selection Guide
            
            **For Regression:**
            - Start with Linear Regression
            - Try KNN for non-linear patterns
            - Use Decision Trees for interpretability
            
            **For Classification:**
            - Start with Logistic Regression
            - Try KNN for complex boundaries
            - Use Decision Trees for rule-based decisions
            
            **For Clustering:**
            - Start with K-Means
            - Experiment with different k values
            - Visualize results when possible
            
            #### 🚀 Advanced Tips
            - **Feature engineering**: Create new features from existing ones
            - **Ensemble methods**: Combine multiple algorithms
            - **Cross-validation**: More robust performance estimates
            - **Regular validation**: Check performance on held-out data
            """)
    
    else:
        st.error("Unable to load algorithm information.")
    
    # Quick actions
    st.markdown("---")
    st.markdown("### 🚀 Ready to Start?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧪 Go to Lab", type="primary", use_container_width=True):
            st.session_state.current_page = "Lab"
            st.rerun()
    
    with col2:
        if st.button("🔬 Compare Algorithms", use_container_width=True):
            st.session_state.current_page = "Compare"
            st.rerun()
    
    with col3:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.current_page = "Dashboard"
            st.rerun()
