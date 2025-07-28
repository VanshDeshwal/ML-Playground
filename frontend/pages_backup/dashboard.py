# Dashboard page implementation
import streamlit as st
import pandas as pd
from ui_components import (
    render_page_header, render_metrics_grid, 
    render_experiment_history_chart, render_algorithm_comparison_chart
)

def render_dashboard_page():
    """Render the analytics dashboard page"""
    render_page_header(
        "Analytics Dashboard", 
        "Track your machine learning experiments and performance insights",
        "📊"
    )
    
    if not st.session_state.experiment_history:
        st.info("🔬 No experiments yet! Go to the Lab to start training models.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 Go to Lab", type="primary", use_container_width=True):
                st.session_state.current_page = "Lab"
                st.rerun()
        
        with col2:
            if st.button("📚 View Documentation", use_container_width=True):
                st.session_state.current_page = "Documentation"
                st.rerun()
        return
    
    # Metrics overview
    successful_experiments = [e for e in st.session_state.experiment_history if e['status'] == 'Success']
    
    if successful_experiments:
        df = pd.DataFrame(successful_experiments)
        
        total_experiments = len(st.session_state.experiment_history)
        success_rate = len(successful_experiments) / total_experiments * 100
        avg_accuracy = df['accuracy'].mean()
        best_accuracy = df['accuracy'].max()
        
        # Metrics grid
        metrics_data = [
            ("Total Experiments", total_experiments, None),
            ("Success Rate", f"{success_rate:.1f}%", None),
            ("Average Score", avg_accuracy, None),
            ("Best Score", best_accuracy, best_accuracy - avg_accuracy if len(successful_experiments) > 1 else None)
        ]
        
        render_metrics_grid(metrics_data)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Performance Over Time")
            render_experiment_history_chart(successful_experiments)
        
        with col2:
            st.markdown("### 🏆 Algorithm Comparison")
            render_algorithm_comparison_chart(successful_experiments)
        
        # Detailed experiment history
        st.markdown("### 📋 Experiment History")
        
        # Filters
        col1, col2 = st.columns([3, 1])
        with col1:
            filter_algorithm = st.selectbox(
                "Filter by Algorithm",
                ["All"] + list(df['algorithm'].unique()),
                key="dashboard_filter"
            )
        
        with col2:
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.experiment_history = []
                st.rerun()
        
        # Filter data
        display_df = df.copy()
        if filter_algorithm != "All":
            display_df = display_df[display_df['algorithm'] == filter_algorithm]
        
        # Display table
        if not display_df.empty:
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df = display_df[['timestamp', 'algorithm', 'accuracy', 'status']].sort_values('timestamp', ascending=False)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": "Time",
                    "algorithm": "Algorithm", 
                    "accuracy": st.column_config.NumberColumn("Score", format="%.4f"),
                    "status": "Status"
                }
            )
        else:
            st.info(f"No experiments found for {filter_algorithm}")
    
    else:
        st.warning("No successful experiments to display metrics.")
        if st.button("🧪 Start Experimenting", type="primary"):
            st.session_state.current_page = "Lab"
            st.rerun()
