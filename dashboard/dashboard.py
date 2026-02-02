import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "alerts_ready.csv")



@st.cache_data
def load_data():
    """Load and cache the alerts data"""
    return pd.read_csv(DATA_PATH)

def display_metrics(df):
    """Show key metrics in the header"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Login Events", f"{len(df):,}")
    with col2:
        high_risk = (df["risk_level"] == "HIGH").sum()
        st.metric("High Risk Alerts", high_risk)
    with col3:
        medium_risk = (df["risk_level"] == "MEDIUM").sum()
        st.metric("Medium Risk Alerts", medium_risk)
    with col4:
        if "Is Account Takeover" in df.columns:
            takeovers = df["Is Account Takeover"].sum()
            st.metric("Known Account Takeovers", takeovers)
        else:
            st.metric("Low Risk Events", (df["risk_level"] == "LOW").sum())

def show_risk_distribution(df):
    """Display risk level and score distributions"""
    st.subheader("📊 Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Risk Level Distribution**")
        risk_counts = df["risk_level"].value_counts()
        st.bar_chart(risk_counts)
    
    with col2:
        st.write("**Risk Score Distribution**")
        fig, ax = plt.subplots(figsize=(8, 4))
        df["risk_score"].hist(bins=30, ax=ax, alpha=0.7)
        ax.set_xlabel("Risk Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Risk Scores")
        st.pyplot(fig)

def show_alert_explorer(df):
    """Interactive alert filtering and exploration"""
    st.subheader("🚨 Alert Explorer")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        risk_filter = st.multiselect(
            "Select Risk Levels",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM"]
        )
    
    with col2:
        min_score = st.slider(
            "Minimum Risk Score", 
            0.0, 1.0, 0.0, 0.1
        )
    
    # Apply filters
    filtered_df = df[
        (df["risk_level"].isin(risk_filter)) & 
        (df["risk_score"] >= min_score)
    ]
    
    st.write(f"Showing {len(filtered_df)} alerts")
    
    # Display results
    display_cols = ["User ID", "risk_score", "risk_level", "explanation"]
    if "Is Account Takeover" in df.columns:
        display_cols.append("Is Account Takeover")
    
    st.dataframe(
        filtered_df[display_cols].sort_values("risk_score", ascending=False),
        use_container_width=True
    )

def show_user_investigation(df):
    """User-specific analysis tool"""
    st.subheader("👤 User Investigation")
    
    # User selection
    user_options = sorted(df["User ID"].unique())
    selected_user = st.selectbox("Select User ID", user_options)
    
    if selected_user:
        user_data = df[df["User ID"] == selected_user]
        
        # User summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Logins", len(user_data))
        with col2:
            avg_risk = user_data["risk_score"].mean()
            st.metric("Average Risk Score", f"{avg_risk:.3f}")
        with col3:
            high_risk_count = (user_data["risk_level"] == "HIGH").sum()
            st.metric("High Risk Events", high_risk_count)
        
        # User timeline
        display_cols = ["Login Timestamp", "risk_score", "risk_level", "explanation"]
        if "Is Account Takeover" in df.columns:
            display_cols.append("Is Account Takeover")
        
        st.dataframe(
            user_data[display_cols].sort_values("Login Timestamp"),
            use_container_width=True
        )

def show_validation_analysis(df):
    """Show model validation against known takeovers"""
    if "Is Account Takeover" not in df.columns:
        return
    
    st.subheader("🎯 Model Validation")
    
    # Risk score comparison
    comparison = df.groupby("Is Account Takeover")["risk_score"].agg(['mean', 'count'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Average Risk Score by Takeover Status**")
        st.bar_chart(comparison['mean'])
    
    with col2:
        st.write("**Detection Statistics**")
        takeover_events = df[df["Is Account Takeover"] == True]
        high_risk_takeovers = takeover_events[takeover_events["risk_level"] == "HIGH"]
        
        if len(takeover_events) > 0:
            detection_rate = len(high_risk_takeovers) / len(takeover_events)
            st.metric("High-Risk Detection Rate", f"{detection_rate:.1%}")
        
        st.write(comparison)

def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="Identity Anomaly Detection",
        page_icon="🔐",
        layout="wide"
    )
    
    st.title("🔐 Identity Anomaly Detection Dashboard")
    st.caption("Behavioral risk analysis using Isolation Forest")
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Alert data not found. Please run the model first.")
        st.stop()
    
    # Main dashboard sections
    display_metrics(df)
    st.divider()
    
    show_risk_distribution(df)
    st.divider()
    
    show_alert_explorer(df)
    st.divider()
    
    show_user_investigation(df)
    st.divider()
    
    show_validation_analysis(df)

if __name__ == "__main__":
    main()
