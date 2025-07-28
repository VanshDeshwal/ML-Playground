# Home page implementation
import streamlit as st
from ui_components import render_page_header, render_algorithm_card, render_status_indicator
from performance_improvements import get_algorithms_cached
from api_utils import get_backend_status

def render_home_page():
    """Render the home/landing page"""
    render_page_header(
        "ML Playground", 
        "Learn, experiment, and compare machine learning algorithms from scratch",
        "🤖"
    )
    
    # Backend status check
    status_info = get_backend_status()
    render_status_indicator(status_info["status"], status_info["message"])
    
    # Quick actions
    st.markdown("""
    <div class="feature-highlight">
        <h3>🚀 Quick Actions</h3>
        <p>Jump right into machine learning experimentation</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧪 Start Experimenting", type="primary", use_container_width=True):
            st.session_state.current_page = "Lab"
            st.rerun()
    
    with col2:
        if st.button("🔬 Compare Algorithms", use_container_width=True):
            st.session_state.current_page = "Compare"
            st.rerun()
    
    with col3:
        if st.button("📚 Learn More", use_container_width=True):
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
        
        # Display algorithms in a grid
        cols = st.columns(3)
        for i, algo in enumerate(algorithms):
            with cols[i % 3]:
                render_algorithm_card(algo)
                if st.button(f"Try {algo['name']}", key=f"try_{algo['id']}", use_container_width=True):
                    st.session_state.selected_algorithm = algo
                    st.session_state.current_page = "Lab"
                    st.rerun()
    else:
        st.warning("Unable to load algorithms. Please check the backend connection.")
    
    # Features overview
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h3>✨ Features</h3>
    </div>
    """, unsafe_allow_html=True)
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        **🔧 From-Scratch Implementations**
        - Pure Python/NumPy implementations
        - Step-by-step learning process
        - Compare with scikit-learn
        
        **📊 Interactive Visualizations**
        - Real-time training progress
        - Performance comparisons
        - Feature importance analysis
        """)
    
    with features_col2:
        st.markdown("""
        **🎯 Multiple Algorithm Types**
        - Regression algorithms
        - Classification methods
        - Clustering techniques
        
        **📈 Experiment Tracking**
        - Performance history
        - Hyperparameter comparison
        - Export results
        """)
