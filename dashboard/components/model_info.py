import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show_model_performance(df):
    """Display model performance metrics and visualizations"""
    st.subheader("🤖 Model Performance & Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Ensemble Architecture**")
        model_weights = {
            'Isolation Forest': 70,
            'One-Class SVM': 20, 
            'Local Outlier Factor': 5,
            'Elliptic Envelope': 5
        }
        
        fig = px.pie(
            values=list(model_weights.values()),
            names=list(model_weights.keys()),
            title="Model Weight Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Detection Performance**")
        if "Is Account Takeover" in df.columns:
            takeover_scores = df[df["Is Account Takeover"] == True]["risk_score"]
            normal_scores = df[df["Is Account Takeover"] == False]["risk_score"]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=takeover_scores, name="Account Takeovers", opacity=0.7))
            fig.add_trace(go.Histogram(x=normal_scores, name="Normal Events", opacity=0.7))
            fig.update_layout(title="Risk Score Distribution", barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)

def show_feature_importance():
    """Display feature importance and takeover-specific indicators"""
    st.subheader("🎯 Feature Engineering & Takeover Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Takeover-Specific Features**")
        takeover_features = {
            'suspicious_combo': 'Device + Country change',
            'failed_then_success': 'Failed then success pattern',
            'night_device_change': 'Night-time device change',
            'high_risk_geo': 'High-risk geographic patterns'
        }
        
        for feature, description in takeover_features.items():
            st.write(f"• **{feature}**: {description}")
    
    with col2:
        st.write("**22 Behavioral Features**")
        feature_categories = {
            'Time Analysis': 4,
            'Session Patterns': 3, 
            'User Profiling': 4,
            'Geographic': 3,
            'Device Intelligence': 2,
            'Takeover Detection': 2,
            'Security Events': 3,
            'Network Analysis': 1
        }
        
        for category, count in feature_categories.items():
            st.write(f"**{category}**: {count} features")

def show_model_metrics(df):
    """Display key model performance metrics"""
    st.subheader("📊 Detection Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if "Is Account Takeover" in df.columns:
            takeover_mean = df[df["Is Account Takeover"] == True]["risk_score"].mean()
            st.metric("Takeover Detection Score", f"{takeover_mean:.3f}")
        else:
            st.metric("High Risk Events", f"{(df['risk_level'] == 'HIGH').sum():,}")
    
    with col2:
        if "Is Account Takeover" in df.columns:
            normal_mean = df[df["Is Account Takeover"] == False]["risk_score"].mean()
            st.metric("Normal Event Score", f"{normal_mean:.3f}")
        else:
            st.metric("Medium Risk Events", f"{(df['risk_level'] == 'MEDIUM').sum():,}")
    
    with col3:
        if "Is Account Takeover" in df.columns:
            takeover_mean = df[df["Is Account Takeover"] == True]["risk_score"].mean()
            normal_mean = df[df["Is Account Takeover"] == False]["risk_score"].mean()
            separation = takeover_mean - normal_mean
            st.metric("Score Separation", f"{separation:.3f}")
        else:
            st.metric("Low Risk Events", f"{(df['risk_level'] == 'LOW').sum():,}")
    
    with col4:
        st.metric("Total Events", f"{len(df):,}")

def show_threshold_config():
    """Display current risk thresholds"""
    st.subheader("⚙️ Risk Thresholds")
    
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config import RISK_THRESHOLDS
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 HIGH", f"{RISK_THRESHOLDS['high']:.2f}")
    with col2:
        st.metric("🟡 MEDIUM", f"{RISK_THRESHOLDS['medium']:.2f}")
    with col3:
        st.metric("🟢 LOW", f"{RISK_THRESHOLDS['low']:.2f}")