import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
import sys
from datetime import datetime

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_THRESHOLDS, FEATURE_WEIGHTS
from src.ueba_visualizations import UEBAVisualizations
from src.ueba_analytics import UserBehavioralAnalytics

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "alerts_ready.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "ensemble_models.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "scaler.pkl")
UEBA_PATH = os.path.join(BASE_DIR, "..", "ueba_data.pkl")



@st.cache_data
def load_data():
    """Load and cache the alerts data"""
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    """Load and cache the trained ensemble models and scaler"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            models = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return models, scaler
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_ueba_data():
    """Load and cache UEBA analytics data"""
    try:
        with open(UEBA_PATH, 'rb') as f:
            ueba_data = pickle.load(f)
        return ueba_data
    except FileNotFoundError:
        return None

def extract_features_from_input(user_input):
    """Extract features from user input for real-time scoring"""
    timestamp = pd.to_datetime(user_input['timestamp'])
    
    features = {
        'login_hour': timestamp.hour,
        'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
        'is_night': 1 if timestamp.hour in [22, 23, 0, 1, 2, 3, 4, 5] else 0,
        'is_business_hours': 1 if 9 <= timestamp.hour < 18 else 0,
        'device_change': user_input.get('device_change', 0),
        'country_change': user_input.get('country_change', 0),
        'high_risk_country': user_input.get('high_risk_country', 0),
        'login_failed': 1 if not user_input.get('success', True) else 0,
        'has_rtt': 1 if user_input.get('rtt') is not None else 0,
        'rtt_vs_global': user_input.get('rtt', 200) - 200,
        'time_since_last_login': user_input.get('time_since_last_login', 24),
        'rapid_login': 1 if user_input.get('time_since_last_login', 24) < 0.1 else 0,
        'hour_deviation': abs(timestamp.hour - user_input.get('typical_hour', 12)),
        'user_login_frequency': user_input.get('user_login_frequency', 1),
        'user_device_diversity': user_input.get('user_device_diversity', 1),
        'user_country_diversity': user_input.get('user_country_diversity', 1),
        # Advanced profiling features
        'weekday_deviation': abs(timestamp.weekday() - 2),  # Assume Tuesday is typical
        'login_consistency': user_input.get('login_consistency', 0.7),
        'user_failure_rate': user_input.get('user_failure_rate', 0.1),
        'user_risk_score': user_input.get('user_risk_score', 0.3),
        # Takeover-specific features
        'suspicious_combo': 1 if (user_input.get('device_change', 0) and user_input.get('country_change', 0)) else 0,
        'failed_then_success': 0  # Cannot determine from single event
    }
    
    return features

def score_single_event(models, scaler, features):
    """Score a single login event using ensemble"""
    import numpy as np
    
    # Ensure features are in the same order as training
    feature_order = [
        "login_hour", "is_weekend", "is_night", "is_business_hours",
        "time_since_last_login", "rapid_login", "hour_deviation", "user_login_frequency",
        "weekday_deviation", "login_consistency", "user_failure_rate", "user_risk_score",
        "device_change", "user_device_diversity", "country_change", "high_risk_country", "user_country_diversity",
        "login_failed", "has_rtt", "rtt_vs_global", "suspicious_combo", "failed_then_success"
    ]
    
    # Create ordered feature array
    ordered_features = [features.get(col, 0) for col in feature_order]
    feature_df = pd.DataFrame([ordered_features], columns=feature_order)
    
    X_scaled = scaler.transform(feature_df)
    
    # Get scores from each model
    scores = {}
    for name, model in models.items():
        if name == 'lof':
            scores[name] = -model.decision_function(X_scaled)[0]
        else:
            scores[name] = -model.decision_function(X_scaled)[0]
    
    # Normalize and ensemble
    weights = {'isolation_forest': 0.4, 'one_class_svm': 0.3, 'lof': 0.2, 'elliptic_envelope': 0.1}
    ensemble_score = sum(weights[name] * max(0, min(1, (score + 1) / 2)) for name, score in scores.items())
    
    if ensemble_score >= RISK_THRESHOLDS["high"]:
        risk_level = "HIGH"
    elif ensemble_score >= RISK_THRESHOLDS["medium"]:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return ensemble_score, risk_level

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

def show_real_time_scorer():
    """Real-time login event scoring interface"""
    st.subheader("🔍 Real-time Login Scorer (Ensemble Models)")
    
    models, scaler = load_model()
    
    if models is None:
        st.error("Ensemble models not found. Please run the training script first.")
        return
    
    st.success(f"Ensemble models loaded successfully! ({len(models)} models)")
    
    # Input form
    with st.form("login_scorer"):
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.text_input("User ID", "user123")
            login_date = st.date_input("Login Date", datetime.now().date())
            login_time = st.time_input("Login Time", datetime.now().time())
            success = st.checkbox("Login Successful", True)
            
        with col2:
            device_change = st.checkbox("New Device Used")
            country_change = st.checkbox("New Country")
            high_risk_country = st.checkbox("High-risk Country")
            rtt = st.number_input("Network Latency (ms)", 0, 2000, 200)
        
        with st.expander("Advanced User Profiling"):
            col3, col4 = st.columns(2)
            with col3:
                time_since_last = st.number_input("Hours Since Last Login", 0.0, 168.0, 24.0)
                typical_hour = st.number_input("User's Typical Login Hour", 0, 23, 12)
                login_consistency = st.slider("User Login Consistency", 0.0, 1.0, 0.7)
            with col4:
                login_frequency = st.number_input("User Login Frequency (per day)", 0.1, 50.0, 2.0)
                user_failure_rate = st.slider("User Historical Failure Rate", 0.0, 1.0, 0.1)
                user_risk_score = st.slider("User Risk Score", 0.0, 1.0, 0.3)
        
        submitted = st.form_submit_button("Score Login Event")
        
        if submitted:
            # Combine date and time
            full_timestamp = datetime.combine(login_date, login_time)
            
            user_input = {
                'user_id': user_id,
                'timestamp': full_timestamp,
                'success': success,
                'device_change': 1 if device_change else 0,
                'country_change': 1 if country_change else 0,
                'high_risk_country': 1 if high_risk_country else 0,
                'rtt': rtt,
                'time_since_last_login': time_since_last,
                'typical_hour': typical_hour,
                'user_login_frequency': login_frequency,
                'user_device_diversity': 2,
                'user_country_diversity': 1,
                'login_consistency': login_consistency,
                'user_failure_rate': user_failure_rate,
                'user_risk_score': user_risk_score
            }
            
            # Extract features and score
            features = extract_features_from_input(user_input)
            risk_score, risk_level = score_single_event(models, scaler, features)
            
            # Display results
            st.subheader("🎯 Ensemble Scoring Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{risk_score:.3f}")
            with col2:
                color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[risk_level]
                st.metric("Risk Level", f"{color} {risk_level}")
            with col3:
                st.metric("User ID", user_id)
            
            # Generate explanation
            reasons = []
            if features['device_change']:
                reasons.append("New device used")
            if features['country_change']:
                reasons.append("New country detected")
            if features['high_risk_country']:
                reasons.append("High-risk location")
            if features['login_failed']:
                reasons.append("Failed login attempt")
            if features['is_night']:
                reasons.append("Night-time login")
            if features['rapid_login']:
                reasons.append("Rapid successive login")
            if features['hour_deviation'] > 6:
                reasons.append("Unusual login time")
            if features.get('user_risk_score', 0) > 0.5:
                reasons.append("High-risk user profile")
            
            explanation = "; ".join(reasons) if reasons else "Behavior within normal range"
            
            st.write("**Explanation:**", explanation)
            
            # Feature breakdown
            with st.expander("Advanced Feature Breakdown"):
                feature_df = pd.DataFrame([features]).T
                feature_df.columns = ['Value']
                st.dataframe(feature_df)

def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="Identity Anomaly Detection",
        page_icon="🔐",
        layout="wide"
    )
    
    st.title("🔐 Identity Anomaly Detection Dashboard")
    st.caption("Behavioral risk analysis using Isolation Forest")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard Overview", "Real-time Scorer", "Alert Explorer", "User Investigation", "UEBA Analytics"]
    )
    
    if page == "Real-time Scorer":
        show_real_time_scorer()
        return
    
    if page == "UEBA Analytics":
        show_ueba_analytics()
        return
    
    # Load data for other pages
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Alert data not found. Please run the model first.")
        st.stop()
    
    if page == "Dashboard Overview":
        display_metrics(df)
        st.divider()
        show_risk_distribution(df)
        st.divider()
        show_validation_analysis(df)
        
    elif page == "Alert Explorer":
        show_alert_explorer(df)
        
    elif page == "User Investigation":
        show_user_investigation(df)

def show_ueba_analytics():
    """UEBA analytics dashboard for SOC analysts"""
    st.subheader("🔍 User & Entity Behavior Analytics (UEBA)")
    
    # Load UEBA data
    ueba_data = load_ueba_data()
    if ueba_data is None:
        st.error("UEBA data not found. Please run the model training first.")
        return
    
    user_baselines = ueba_data['user_baselines']
    risk_progressions = ueba_data['risk_progressions']
    ueba_analytics = ueba_data['ueba_analytics']
    
    # Load main data
    df = load_data()
    
    # Initialize visualizations
    viz = UEBAVisualizations()
    
    # Analysis mode selection
    analysis_mode = st.radio(
        "Analysis Mode",
        ["High-Risk Users Overview", "Individual User Deep-Dive"],
        horizontal=True
    )
    
    if analysis_mode == "High-Risk Users Overview":
        st.subheader("🚨 High-Risk Users Dashboard")
        
        # Create risk ranking
        risk_ranking = []
        for user_id, progression in risk_progressions.items():
            summary = ueba_analytics.generate_user_risk_summary(
                user_id, df, user_baselines, risk_progressions
            )
            if summary:
                risk_ranking.append({
                    'User ID': user_id,
                    'Current Risk Score': summary['current_risk_score'],
                    'Risk Level': summary['current_risk_level'],
                    'Risk Trend': summary['risk_trend'],
                    'Profile Type': summary['profile_type'],
                    'Recent Escalations': summary['recent_escalations'],
                    'Recommendation': summary['recommendation'],
                    'Behavioral Flags': ', '.join(summary['behavioral_flags']) if summary['behavioral_flags'] else 'None'
                })
        
        risk_df = pd.DataFrame(risk_ranking)
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            min_risk_score = st.slider("Minimum Risk Score", 0.0, 1.0, 0.3, 0.1)
        with col2:
            risk_levels = st.multiselect(
                "Risk Levels", 
                ["HIGH", "MEDIUM", "LOW"], 
                default=["HIGH", "MEDIUM"]
            )
        with col3:
            min_trend = st.slider("Minimum Risk Trend", -0.1, 0.2, 0.0, 0.01)
        
        # Apply filters
        filtered_df = risk_df[
            (risk_df['Current Risk Score'] >= min_risk_score) &
            (risk_df['Risk Level'].isin(risk_levels)) &
            (risk_df['Risk Trend'] >= min_trend)
        ].sort_values(['Risk Trend', 'Current Risk Score'], ascending=[False, False])
        
        st.write(f"**{len(filtered_df)} users match criteria**")
        
        # Show top users table
        st.dataframe(
            filtered_df.head(20),
            use_container_width=True,
            column_config={
                "Current Risk Score": st.column_config.ProgressColumn(
                    "Risk Score",
                    help="Current risk score",
                    min_value=0,
                    max_value=1,
                ),
                "Risk Trend": st.column_config.NumberColumn(
                    "Trend",
                    help="Risk trend (positive = increasing)",
                    format="%.3f"
                )
            }
        )
        
        # Show timelines for top 3 users
        if len(filtered_df) > 0:
            st.subheader("📈 Top 3 Risk Timelines")
            
            top_users = filtered_df.head(3)['User ID'].tolist()
            
            for i, user_id in enumerate(top_users):
                with st.expander(f"User {user_id} - Risk Timeline", expanded=(i==0)):
                    # Show user summary
                    user_summary = filtered_df[filtered_df['User ID'] == user_id].iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Risk Score", f"{user_summary['Current Risk Score']:.3f}")
                    with col2:
                        st.metric("Risk Level", user_summary['Risk Level'])
                    with col3:
                        st.metric("Trend", f"{user_summary['Risk Trend']:.3f}")
                    with col4:
                        st.metric("Profile", user_summary['Profile Type'])
                    
                    if user_summary['Behavioral Flags'] != 'None':
                        st.warning(f"**Flags:** {user_summary['Behavioral Flags']}")
                    
                    st.info(f"**Recommendation:** {user_summary['Recommendation']}")
                    
                    # Show timeline
                    viz.show_user_risk_timeline(user_id, risk_progressions)
    
    else:  # Individual User Deep-Dive
        st.subheader("🔍 Individual User Analysis")
        
        # User selection with search
        available_users = list(risk_progressions.keys())
        
        # Show top risky users as suggestions
        risk_ranking = []
        for user_id, progression in risk_progressions.items():
            risk_ranking.append({
                'user_id': user_id,
                'risk_score': progression.get('max_risk_score', 0),
                'trend': progression.get('current_trend', 0)
            })
        
        top_risky = sorted(risk_ranking, key=lambda x: (x['trend'], x['risk_score']), reverse=True)[:10]
        
        st.write("**Suggested High-Risk Users:**")
        suggestion_cols = st.columns(5)
        for i, user_info in enumerate(top_risky[:5]):
            with suggestion_cols[i]:
                if st.button(f"User {user_info['user_id']}", key=f"suggest_{i}"):
                    st.session_state.selected_user = user_info['user_id']
        
        # User selection dropdown
        selected_user = st.selectbox(
            "Or select any user:", 
            available_users,
            index=0 if 'selected_user' not in st.session_state else available_users.index(st.session_state.get('selected_user', available_users[0]))
        )
        
        # Show detailed analysis for selected user
        summary = ueba_analytics.generate_user_risk_summary(
            selected_user, df, user_baselines, risk_progressions
        )
        
        if summary:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Profile Type", summary['profile_type'])
            with col2:
                st.metric("Current Risk", summary['current_risk_level'])
            with col3:
                st.metric("Risk Trend", f"{summary['risk_trend']:.3f}")
            with col4:
                st.metric("Stability", f"{summary['stability_score']:.3f}")
            
            st.info(f"**Recommendation:** {summary['recommendation']}")
            
            if summary['behavioral_flags']:
                st.warning(f"**Behavioral Flags:** {', '.join(summary['behavioral_flags'])}")
        
        # Show timeline visualization
        viz.show_user_risk_timeline(selected_user, risk_progressions)

if __name__ == "__main__":
    main()
