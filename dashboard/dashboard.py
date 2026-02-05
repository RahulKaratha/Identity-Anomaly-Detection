import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modular components
from components.model_info import show_model_performance, show_feature_importance, show_model_metrics, show_threshold_config
from components.scorer import show_real_time_scorer
from components.data_utils import (load_alerts_data, display_key_metrics, show_alert_explorer, 
                                  show_user_investigation, show_risk_distribution_charts, 
                                  show_detection_metrics_charts, show_takeover_detection_summary)
import plotly.express as px
import plotly.graph_objects as go


def show_ueba_analytics():
    """Enhanced UEBA analytics with detailed event capture and better visualization"""
    st.subheader("🔍 User & Entity Behavior Analytics")
    
    try:
        df = load_alerts_data()
    except:
        st.error("❌ Alert data not found. Run model training first.")
        return
    
    if 'User ID' not in df.columns:
        st.error("❌ User ID column not found in data.")
        return
    
    st.info("📊 Analyzing user behavior patterns with enhanced event details...")
    
    # Enhanced analysis with better user selection
    with st.spinner("Finding users with increasing risk trends..."):
        # Find users with clear increasing trends
        increasing_users = []
        
        for user_id in df['User ID'].unique():
            user_data = df[df['User ID'] == user_id].copy()
            if len(user_data) >= 5:  # Need sufficient events
                if 'Login Timestamp' in user_data.columns:
                    user_data['Login Timestamp'] = pd.to_datetime(user_data['Login Timestamp'])
                    user_data = user_data.sort_values('Login Timestamp')
                
                # Calculate trend using linear regression
                x = np.arange(len(user_data))
                y = user_data['risk_score'].values
                trend_slope = np.polyfit(x, y, 1)[0]
                
                # Only include users with significant upward trend
                if trend_slope > 0.01:  # Meaningful increase
                    risk_increase = user_data['risk_score'].iloc[-1] - user_data['risk_score'].iloc[0]
                    increasing_users.append({
                        'User ID': user_id,
                        'Events': len(user_data),
                        'Trend_Slope': trend_slope,
                        'Risk_Increase': risk_increase,
                        'Start_Risk': user_data['risk_score'].iloc[0],
                        'End_Risk': user_data['risk_score'].iloc[-1],
                        'Max_Risk': user_data['risk_score'].max(),
                        'High_Risk_Events': (user_data['risk_level'] == 'HIGH').sum(),
                        'Countries': len(user_data['Country'].unique()) if 'Country' in user_data.columns else 1,
                        'Devices': len(user_data['Device Type'].unique()) if 'Device Type' in user_data.columns else 1
                    })
        
        trend_df = pd.DataFrame(increasing_users)
        top_increasing = trend_df.sort_values('Trend_Slope', ascending=False).head(5)
    
    if len(top_increasing) == 0:
        st.warning("⚠️ No users found with significant increasing risk trends.")
        return
    
    st.success(f"✅ Found {len(top_increasing)} users with increasing risk trends")
    
    # Display summary table with enhanced metrics
    st.subheader("📊 Top 5 Users with Increasing Risk Trends")
    display_df = top_increasing[['User ID', 'Events', 'Start_Risk', 'End_Risk', 'Risk_Increase', 'High_Risk_Events', 'Countries', 'Devices']].copy()
    display_df['Start_Risk'] = display_df['Start_Risk'].round(3)
    display_df['End_Risk'] = display_df['End_Risk'].round(3)
    display_df['Risk_Increase'] = display_df['Risk_Increase'].round(3)
    st.dataframe(display_df, use_container_width=True)
    
    # Enhanced individual user analysis
    st.subheader("📈 Detailed Risk Progression Analysis")
    
    for idx, (_, user_row) in enumerate(top_increasing.iterrows()):
        user_id = user_row['User ID']
        
        with st.expander(f"👤 User {user_id} - Risk Trend Analysis", expanded=(idx < 2)):
            user_data = df[df['User ID'] == user_id].copy()
            
            if 'Login Timestamp' in user_data.columns:
                user_data['Login Timestamp'] = pd.to_datetime(user_data['Login Timestamp'])
                user_data = user_data.sort_values('Login Timestamp')
            
            user_data['Event_Number'] = range(1, len(user_data) + 1)
            
            # Enhanced metrics display
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Events", len(user_data))
            with col2:
                risk_increase = user_data['risk_score'].iloc[-1] - user_data['risk_score'].iloc[0]
                st.metric("Risk Increase", f"{risk_increase:+.3f}", delta=f"{risk_increase:+.3f}")
            with col3:
                st.metric("Current Risk", f"{user_data['risk_score'].iloc[-1]:.3f}")
            with col4:
                high_risk_count = (user_data['risk_level'] == 'HIGH').sum()
                st.metric("High Risk Events", high_risk_count)
            with col5:
                countries = len(user_data['Country'].unique()) if 'Country' in user_data.columns else 1
                st.metric("Countries", countries)
            
            # Enhanced visualization with better scale and event details
            fig = go.Figure()
            
            # Create detailed hover information
            hover_text = []
            for _, row in user_data.iterrows():
                hover_info = f"""<b>Event #{row['Event_Number']}</b><br>
                <b>Time:</b> {row.get('Login Timestamp', 'Unknown')}<br>
                <b>Risk Score:</b> {row['risk_score']:.3f}<br>
                <b>Risk Level:</b> {row['risk_level']}<br>
                <b>Country:</b> {row.get('Country', 'Unknown')}<br>
                <b>Device:</b> {row.get('Device Type', 'Unknown')}<br>
                <b>IP:</b> {row.get('IP Address', 'Unknown')}<br>
                <b>Success:</b> {'Yes' if row.get('Login Successful', True) else 'No'}<br>
                <b>Explanation:</b> {row.get('explanation', 'No explanation')[:80]}..."""
                hover_text.append(hover_info)
            
            # Color points by risk level
            colors = {'HIGH': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#44ff44'}
            point_colors = [colors.get(level, '#cccccc') for level in user_data['risk_level']]
            
            # Main risk progression line with enhanced markers
            fig.add_trace(go.Scatter(
                x=user_data['Event_Number'],
                y=user_data['risk_score'],
                mode='lines+markers',
                name='Risk Score',
                line=dict(color='#1f77b4', width=3),
                marker=dict(
                    size=10,
                    color=point_colors,
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover_text
            ))
            
            # Add trend line
            x_vals = np.arange(len(user_data))
            trend_line = np.polyfit(x_vals, user_data['risk_score'], 1)
            trend_y = np.polyval(trend_line, x_vals)
            
            fig.add_trace(go.Scatter(
                x=user_data['Event_Number'],
                y=trend_y,
                mode='lines',
                name=f'Trend (slope: {trend_line[0]:.4f})',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='Trend Line<extra></extra>'
            ))
            
            # Highlight high-risk events
            high_risk_events = user_data[user_data['risk_level'] == 'HIGH']
            if len(high_risk_events) > 0:
                fig.add_trace(go.Scatter(
                    x=high_risk_events['Event_Number'],
                    y=high_risk_events['risk_score'],
                    mode='markers',
                    name='High Risk Events',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='diamond',
                        line=dict(width=3, color='darkred')
                    ),
                    hovertemplate='<b>HIGH RISK EVENT</b><br>%{customdata}<extra></extra>',
                    customdata=[hover_text[i] for i, idx in enumerate(user_data.index) if idx in high_risk_events.index]
                ))
            
            # Add threshold lines with better visibility
            fig.add_hline(y=0.75, line_dash="dash", line_color="red", line_width=2,
                         annotation_text="High Risk (0.75)", annotation_position="top right")
            fig.add_hline(y=0.6, line_dash="dash", line_color="orange", line_width=2,
                         annotation_text="Medium Risk (0.6)", annotation_position="top right")
            
            # Improved layout with better scale
            min_risk = max(0, user_data['risk_score'].min() - 0.05)
            max_risk = min(1, user_data['risk_score'].max() + 0.05)
            
            fig.update_layout(
                title=f"Risk Progression: User {user_id} (Trend: {trend_line[0]:+.4f} per event)",
                xaxis_title="Event Number (Chronological Order)",
                yaxis_title="Risk Score",
                yaxis=dict(range=[min_risk, max_risk]),  # Dynamic scale for better visibility
                hovermode='closest',
                showlegend=True,
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Event details table with enhanced filtering
            st.write("**Recent Event Details:**")
            
            # Show last 10 events with key details
            recent_events = user_data.tail(10)
            display_cols = ['Event_Number', 'risk_score', 'risk_level', 'Country', 'Device Type', 'IP Address', 'Login Successful', 'explanation']
            available_cols = [col for col in display_cols if col in recent_events.columns]
            
            # Format the display
            display_events = recent_events[available_cols].copy()
            if 'risk_score' in display_events.columns:
                display_events['risk_score'] = display_events['risk_score'].round(3)
            
            st.dataframe(display_events, use_container_width=True, height=300)
    
    # Additional insights section
    st.subheader("🔍 Key Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**🚨 Alert Summary:**\n- {len(top_increasing)} users show increasing risk trends\n- Average risk increase: {top_increasing['Risk_Increase'].mean():.3f}\n- Total high-risk events: {top_increasing['High_Risk_Events'].sum()}")
    
    with col2:
        st.warning(f"**⚠️ Risk Factors:**\n- Multi-country access: {(top_increasing['Countries'] > 1).sum()} users\n- Multi-device usage: {(top_increasing['Devices'] > 1).sum()} users\n- Highest risk score: {top_increasing['Max_Risk'].max():.3f}")
    
    # Quick user selector for detailed analysis
    st.subheader("🔎 Quick User Analysis")
    selected_user = st.selectbox(
        "Select a user for detailed analysis:",
        options=top_increasing['User ID'].tolist(),
        help="Choose from users with increasing risk trends"
    )
    
    if selected_user:
        user_data = df[df['User ID'] == selected_user].copy()
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Events", len(user_data))
        with col2:
            unique_countries = len(user_data['Country'].unique()) if 'Country' in user_data.columns else 1
            st.metric("Countries", unique_countries)
        with col3:
            failed_logins = (user_data.get('Login Successful', True) == False).sum()
            st.metric("Failed Logins", failed_logins)
        with col4:
            avg_risk = user_data['risk_score'].mean()
            st.metric("Avg Risk", f"{avg_risk:.3f}")
        
        # Show all events for selected user
        st.write("**All Events for Selected User:**")
        display_cols = ['User ID', 'Login Timestamp', 'risk_score', 'risk_level', 'Country', 'Device Type', 'IP Address', 'Login Successful', 'explanation']
        available_cols = [col for col in display_cols if col in user_data.columns]
        
        # Sort by timestamp if available
        if 'Login Timestamp' in user_data.columns:
            user_data['Login Timestamp'] = pd.to_datetime(user_data['Login Timestamp'])
            user_data = user_data.sort_values('Login Timestamp')
        
        display_data = user_data[available_cols].copy()
        if 'risk_score' in display_data.columns:
            display_data['risk_score'] = display_data['risk_score'].round(3)
        
        st.dataframe(display_data, use_container_width=True, height=400)
    
    # Legacy individual deep-dive mode (simplified)
    with st.expander("🔬 Advanced Individual Analysis"):
        st.write("Select any user from the dataset for comprehensive analysis:")
        all_users = sorted(df['User ID'].unique())
        deep_dive_user = st.selectbox("Choose user:", all_users, key="deep_dive")
        available_users = sorted(df['User ID'].unique())
        selected_user = st.selectbox("Select User", available_users)
        
        if deep_dive_user:
            user_data = df[df['User ID'] == deep_dive_user].copy()
            
            # Comprehensive user analysis
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Events", len(user_data))
            with col2:
                avg_risk = user_data['risk_score'].mean()
                st.metric("Avg Risk Score", f"{avg_risk:.3f}")
            with col3:
                high_risk_count = (user_data['risk_level'] == 'HIGH').sum()
                st.metric("High Risk Events", high_risk_count)
            with col4:
                max_risk = user_data['risk_score'].max()
                st.metric("Max Risk Score", f"{max_risk:.3f}")
            
            # Show all events for comprehensive analysis
            st.write("**Complete Event History:**")
            display_cols = ['Login Timestamp', 'risk_score', 'risk_level', 'Country', 'Device Type', 'IP Address', 'Login Successful', 'explanation']
            available_cols = [col for col in display_cols if col in user_data.columns]
            
            if 'Login Timestamp' in user_data.columns:
                user_data['Login Timestamp'] = pd.to_datetime(user_data['Login Timestamp'])
                user_data = user_data.sort_values('Login Timestamp')
            
            display_data = user_data[available_cols].copy()
            if 'risk_score' in display_data.columns:
                display_data['risk_score'] = display_data['risk_score'].round(3)
            
            st.dataframe(display_data, use_container_width=True, height=400)

def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="UEBA Identity Anomaly Detection",
        page_icon="🔐",
        layout="wide"
    )
    
    st.title("🔐 Advanced UEBA Identity Anomaly Detection")
    st.caption("Enterprise-grade account takeover detection with ensemble ML models")
    
    # Navigation
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard Overview", "Model Performance", "Real-time Scorer", "Alert Explorer", "User Investigation", "UEBA Analytics"]
    )
    
    # Load data for most pages
    if page not in ["Real-time Scorer"]:
        try:
            df = load_alerts_data()
        except FileNotFoundError:
            st.error("❌ Alert data not found. Run model training first.")
            st.stop()
    
    # Page routing
    if page == "Dashboard Overview":
        display_key_metrics(df)
        st.divider()
        show_risk_distribution_charts(df)
        st.divider()
        show_detection_metrics_charts(df)
        st.divider()
        show_takeover_detection_summary(df)
        st.divider()
        show_model_metrics(df)
        st.divider()
        show_threshold_config()
        
    elif page == "Model Performance":
        show_model_performance(df)
        st.divider()
        show_feature_importance()
        
    elif page == "Real-time Scorer":
        show_real_time_scorer()
        
    elif page == "Alert Explorer":
        show_alert_explorer(df)
        
    elif page == "User Investigation":
        show_user_investigation(df)
        
    elif page == "UEBA Analytics":
        show_ueba_analytics()

if __name__ == "__main__":
    main()
