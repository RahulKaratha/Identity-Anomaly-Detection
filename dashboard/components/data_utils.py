import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

def get_base_dir():
    """Get the project base directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_dir = os.path.dirname(current_dir)  # dashboard/
    return os.path.dirname(dashboard_dir)  # project root

@st.cache_data
def load_alerts_data():
    """Load and cache alerts data"""
    BASE_DIR = get_base_dir()
    DATA_PATH = os.path.join(BASE_DIR, "data", "alerts_ready.csv")
    
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found: {DATA_PATH}")
        st.info(f"Current working directory: {os.getcwd()}")
        st.info(f"Base directory: {BASE_DIR}")
        raise FileNotFoundError(f"Alert data not found at {DATA_PATH}")
    
    return pd.read_csv(DATA_PATH)

def display_key_metrics(df):
    """Display enhanced key metrics with detection rates"""
    col1, col2, col3, col4 = st.columns(4)
    
    total_events = len(df)
    high_risk = (df["risk_level"] == "HIGH").sum()
    medium_risk = (df["risk_level"] == "MEDIUM").sum()
    low_risk = (df["risk_level"] == "LOW").sum()
    
    with col1:
        st.metric("Total Events", f"{total_events:,}")
    
    with col2:
        high_pct = (high_risk / total_events * 100) if total_events > 0 else 0
        st.metric("🔴 High Risk", f"{high_risk:,}", f"{high_pct:.1f}%")
    
    with col3:
        medium_pct = (medium_risk / total_events * 100) if total_events > 0 else 0
        st.metric("🟡 Medium Risk", f"{medium_risk:,}", f"{medium_pct:.1f}%")
    
    with col4:
        if "Is Account Takeover" in df.columns:
            takeovers = df["Is Account Takeover"].sum()
            takeover_pct = (takeovers / total_events * 100) if total_events > 0 else 0
            st.metric("🎯 Known Takeovers", f"{takeovers:,}", f"{takeover_pct:.2f}%")
        else:
            low_pct = (low_risk / total_events * 100) if total_events > 0 else 0
            st.metric("🟢 Low Risk", f"{low_risk:,}", f"{low_pct:.1f}%")
    
    # Add detection effectiveness metrics
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_risk_score = df["risk_score"].mean()
        st.metric("Avg Risk Score", f"{avg_risk_score:.3f}")
    
    with col2:
        if "Is Account Takeover" in df.columns:
            # Calculate detection rate for known takeovers
            takeover_df = df[df["Is Account Takeover"] == True]
            if len(takeover_df) > 0:
                detected_takeovers = ((takeover_df["risk_level"] == "HIGH") | 
                                    (takeover_df["risk_level"] == "MEDIUM")).sum()
                detection_rate = (detected_takeovers / len(takeover_df) * 100)
                st.metric("Takeover Detection Rate", f"{detection_rate:.1f}%")
            else:
                st.metric("Takeover Detection Rate", "N/A")
        else:
            # Show high-risk detection efficiency
            high_risk_avg = df[df["risk_level"] == "HIGH"]["risk_score"].mean() if high_risk > 0 else 0
            st.metric("High Risk Avg Score", f"{high_risk_avg:.3f}")
    
    with col3:
        # False positive rate (assuming low risk scores should be truly low risk)
        if "Is Account Takeover" in df.columns:
            normal_df = df[df["Is Account Takeover"] == False]
            if len(normal_df) > 0:
                false_positives = (normal_df["risk_level"] == "HIGH").sum()
                fp_rate = (false_positives / len(normal_df) * 100)
                st.metric("False Positive Rate", f"{fp_rate:.2f}%")
            else:
                st.metric("False Positive Rate", "N/A")
        else:
            # Show medium risk efficiency
            medium_risk_avg = df[df["risk_level"] == "MEDIUM"]["risk_score"].mean() if medium_risk > 0 else 0
            st.metric("Medium Risk Avg Score", f"{medium_risk_avg:.3f}")
    
    with col4:
        # Risk score distribution efficiency
        risk_std = df["risk_score"].std()
        st.metric("Risk Score Std Dev", f"{risk_std:.3f}")

def show_alert_explorer(df):
    """Interactive alert filtering"""
    st.subheader("🚨 Alert Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_filter = st.multiselect(
            "Risk Levels",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM"]
        )
    
    with col2:
        min_score = st.slider("Min Risk Score", 0.0, 1.0, 0.0, 0.1)
    
    filtered_df = df[
        (df["risk_level"].isin(risk_filter)) & 
        (df["risk_score"] >= min_score)
    ]
    
    st.write(f"**{len(filtered_df)} alerts match criteria**")
    
    display_cols = ["User ID", "risk_score", "risk_level", "explanation"]
    if "Is Account Takeover" in df.columns:
        display_cols.append("Is Account Takeover")
    
    st.dataframe(
        filtered_df[display_cols].sort_values("risk_score", ascending=False).head(100),
        use_container_width=True
    )

def show_user_investigation(df):
    """User-specific analysis"""
    st.subheader("👤 User Investigation")
    
    user_options = sorted(df["User ID"].unique())
    selected_user = st.selectbox("Select User", user_options)
    
    if selected_user:
        user_data = df[df["User ID"] == selected_user]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Logins", len(user_data))
        with col2:
            avg_risk = user_data["risk_score"].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.3f}")
        with col3:
            high_risk_count = (user_data["risk_level"] == "HIGH").sum()
            st.metric("High Risk Events", high_risk_count)
        
        display_cols = ["Login Timestamp", "risk_score", "risk_level", "explanation"]
        if "Is Account Takeover" in df.columns:
            display_cols.append("Is Account Takeover")
        
        st.dataframe(
            user_data[display_cols].sort_values("Login Timestamp"),
            use_container_width=True
        )
        
def show_risk_distribution_charts(df):
    """Display risk distribution visualizations"""
    st.subheader("📈 Risk Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk level bar chart
        risk_counts = df['risk_level'].value_counts()
        colors = {'HIGH': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#44ff44'}
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker_color=[colors.get(level, '#cccccc') for level in risk_counts.index],
                text=risk_counts.values,
                textposition='auto',
            )
        ])
        
        fig_bar.update_layout(
            title="Alert Distribution by Risk Level",
            xaxis_title="Risk Level",
            yaxis_title="Number of Alerts",
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Risk score distribution histogram
        fig_hist = px.histogram(
            df, 
            x='risk_score', 
            nbins=50,
            title="Risk Score Distribution",
            color_discrete_sequence=['#1f77b4']
        )
        
        # Add threshold lines
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        try:
            from config import RISK_THRESHOLDS
            fig_hist.add_vline(x=RISK_THRESHOLDS['high'], line_dash="dash", line_color="red", annotation_text="High")
            fig_hist.add_vline(x=RISK_THRESHOLDS['medium'], line_dash="dash", line_color="orange", annotation_text="Medium")
        except:
            pass
        
        fig_hist.update_layout(
            xaxis_title="Risk Score",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)

def show_detection_metrics_charts(df):
    """Display detection effectiveness charts"""
    st.subheader("🎯 Detection Effectiveness")
    
    if "Is Account Takeover" in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Takeover detection by risk level
            takeover_df = df[df["Is Account Takeover"] == True]
            normal_df = df[df["Is Account Takeover"] == False]
            
            if len(takeover_df) > 0 and len(normal_df) > 0:
                takeover_risk_dist = takeover_df['risk_level'].value_counts()
                normal_risk_dist = normal_df['risk_level'].value_counts()
                
                fig_detection = go.Figure()
                
                fig_detection.add_trace(go.Bar(
                    name='Account Takeovers',
                    x=takeover_risk_dist.index,
                    y=takeover_risk_dist.values,
                    marker_color='red',
                    opacity=0.7
                ))
                
                fig_detection.add_trace(go.Bar(
                    name='Normal Events',
                    x=normal_risk_dist.index,
                    y=normal_risk_dist.values,
                    marker_color='blue',
                    opacity=0.7
                ))
                
                fig_detection.update_layout(
                    title="Risk Level Distribution: Takeovers vs Normal",
                    xaxis_title="Risk Level",
                    yaxis_title="Count",
                    barmode='group'
                )
                
                st.plotly_chart(fig_detection, use_container_width=True)
        
        with col2:
            # ROC-like curve showing score separation
            takeover_scores = takeover_df['risk_score'].values
            normal_scores = normal_df['risk_score'].values
            
            fig_scores = go.Figure()
            
            fig_scores.add_trace(go.Histogram(
                x=normal_scores,
                name='Normal Events',
                opacity=0.7,
                marker_color='blue',
                nbinsx=30
            ))
            
            fig_scores.add_trace(go.Histogram(
                x=takeover_scores,
                name='Account Takeovers',
                opacity=0.7,
                marker_color='red',
                nbinsx=30
            ))
            
            fig_scores.update_layout(
                title="Risk Score Distribution Comparison",
                xaxis_title="Risk Score",
                yaxis_title="Frequency",
                barmode='overlay'
            )
            
            st.plotly_chart(fig_scores, use_container_width=True)
    
    else:
        # Show risk score trends over time if timestamp available
        if 'Login Timestamp' in df.columns:
            df_time = df.copy()
            df_time['Login Timestamp'] = pd.to_datetime(df_time['Login Timestamp'])
            df_time['Date'] = df_time['Login Timestamp'].dt.date
            
            daily_risk = df_time.groupby(['Date', 'risk_level']).size().unstack(fill_value=0)
            
            fig_trend = go.Figure()
            
            colors = {'HIGH': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#44ff44'}
            for risk_level in ['HIGH', 'MEDIUM', 'LOW']:
                if risk_level in daily_risk.columns:
                    fig_trend.add_trace(go.Scatter(
                        x=daily_risk.index,
                        y=daily_risk[risk_level],
                        mode='lines+markers',
                        name=f'{risk_level} Risk',
                        line=dict(color=colors[risk_level])
                    ))
            
            fig_trend.update_layout(
                title="Daily Risk Alert Trends",
                xaxis_title="Date",
                yaxis_title="Number of Alerts"
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)

def show_takeover_detection_summary(df):
    """Display takeover detection summary with percentages"""
    if "Is Account Takeover" in df.columns:
        st.subheader("🔍 Takeover Detection Summary")
        
        takeover_df = df[df["Is Account Takeover"] == True]
        total_takeovers = len(takeover_df)
        
        if total_takeovers > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                high_detected = (takeover_df["risk_level"] == "HIGH").sum()
                high_pct = (high_detected / total_takeovers * 100)
                st.metric("High Risk Detected", f"{high_detected}/{total_takeovers}", f"{high_pct:.1f}%")
            
            with col2:
                medium_detected = (takeover_df["risk_level"] == "MEDIUM").sum()
                medium_pct = (medium_detected / total_takeovers * 100)
                st.metric("Medium Risk Detected", f"{medium_detected}/{total_takeovers}", f"{medium_pct:.1f}%")
            
            with col3:
                total_detected = high_detected + medium_detected
                total_pct = (total_detected / total_takeovers * 100)
                st.metric("Total Detected", f"{total_detected}/{total_takeovers}", f"{total_pct:.1f}%")
            
            with col4:
                missed = total_takeovers - total_detected
                missed_pct = (missed / total_takeovers * 100)
                st.metric("Missed (Low Risk)", f"{missed}/{total_takeovers}", f"{missed_pct:.1f}%")
        else:
            st.info("No takeover data available for analysis.")