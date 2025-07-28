# UI Components and widgets
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from performance_improvements import create_metric_card_optimized, optimize_plotly_config

def render_sidebar_navigation():
    """Render the main navigation sidebar"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 2rem;">
            <h2 style="margin: 0;">🤖 ML Playground</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Explore Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons
        pages = {
            "🏠 Home": "Home",
            "🧪 Lab": "Lab", 
            "🔬 Compare": "Compare",
            "📊 Dashboard": "Dashboard",
            "📚 Documentation": "Documentation"
        }
        
        st.markdown("### Navigation")
        for label, page in pages.items():
            if st.button(label, key=f"nav_{page}", use_container_width=True):
                st.session_state.current_page = page
                st.rerun()

def render_algorithm_card(algorithm):
    """Render a single algorithm card"""
    icon_map = {
        'regression': '📈',
        'classification': '🎯', 
        'clustering': '🧩'
    }
    
    icon = icon_map.get(algorithm['type'], '🔧')
    
    st.markdown(f"""
    <div class="algorithm-card">
        <div class="algorithm-icon">{icon}</div>
        <h4>{algorithm['name']}</h4>
        <p><strong>Type:</strong> {algorithm['type'].title()}</p>
        <p>{algorithm['description']}</p>
    </div>
    """, unsafe_allow_html=True)

def render_metrics_grid(metrics_data):
    """Render metrics in a responsive grid"""
    cols = st.columns(len(metrics_data))
    for i, (title, value, delta) in enumerate(metrics_data):
        with cols[i]:
            st.markdown(
                create_metric_card_optimized(title, value, delta), 
                unsafe_allow_html=True
            )

def render_experiment_history_chart(experiment_data):
    """Render performance chart with optimization"""
    if not experiment_data:
        st.info("No experiment data available yet.")
        return
    
    import pandas as pd
    df = pd.DataFrame(experiment_data)
    df['experiment_id'] = range(len(df), 0, -1)
    
    fig = px.line(df, x='experiment_id', y='accuracy', color='algorithm',
                 title="Model Performance Over Time",
                 labels={'experiment_id': 'Experiment Number', 'accuracy': 'Score'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, config=optimize_plotly_config())

def render_algorithm_comparison_chart(experiment_data):
    """Render algorithm comparison bar chart"""
    if not experiment_data:
        return
    
    import pandas as pd
    df = pd.DataFrame(experiment_data)
    algo_avg = df.groupby('algorithm')['accuracy'].mean().to_dict()
    
    fig = px.bar(x=list(algo_avg.keys()), y=list(algo_avg.values()),
                title="Average Performance by Algorithm",
                labels={'x': 'Algorithm', 'y': 'Average Score'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, config=optimize_plotly_config())

def render_status_indicator(status, message):
    """Render backend status indicator"""
    if status == "healthy":
        st.success(f"✅ {message}")
    else:
        st.error(f"❌ {message}")

def render_page_header(title, description, icon="🤖"):
    """Render consistent page headers"""
    st.markdown(f"""
    <div class="main-header">
        <h1>{icon} {title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)
