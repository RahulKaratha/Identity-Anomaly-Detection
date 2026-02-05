import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import plotly.graph_objects as go

def load_models():
    """Load ensemble models and scaler"""
    # Get project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_dir = os.path.dirname(current_dir)  # dashboard/
    BASE_DIR = os.path.dirname(dashboard_dir)  # project root
    
    MODEL_PATH = os.path.join(BASE_DIR, "ensemble_models.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            models = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return models, scaler
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info(f"Looking for models at: {MODEL_PATH}")
        return None, None

def extract_features(user_input):
    """Extract 22 behavioral features from user input with enhanced anomaly detection"""
    timestamp = pd.to_datetime(user_input['timestamp'])
    
    # Enhanced suspicious combination detection
    device_change = user_input.get('device_change', 0)
    country_change = user_input.get('country_change', 0)
    high_risk_country = user_input.get('high_risk_country', 0)
    login_failed = 1 if not user_input.get('success', True) else 0
    
    # More sophisticated suspicious combo calculation
    suspicious_combo = 0
    if device_change and country_change:
        suspicious_combo = 1
    elif device_change and high_risk_country:
        suspicious_combo = 1
    elif country_change and login_failed:
        suspicious_combo = 1
    
    # Time-based anomaly detection
    hour = timestamp.hour
    is_night = 1 if hour in [22, 23, 0, 1, 2, 3, 4, 5] else 0
    is_weekend = 1 if timestamp.weekday() >= 5 else 0
    is_business_hours = 1 if 9 <= hour < 18 else 0
    
    # Enhanced time deviation calculation
    typical_hour = user_input.get('typical_hour', 12)
    hour_deviation = min(abs(hour - typical_hour), 24 - abs(hour - typical_hour))
    
    # Network latency analysis
    rtt = user_input.get('rtt', 200)
    rtt_vs_global = rtt - 200
    
    return {
        'login_hour': hour,
        'is_weekend': is_weekend,
        'is_night': is_night,
        'is_business_hours': is_business_hours,
        'time_since_last_login': user_input.get('time_since_last_login', 24),
        'rapid_login': 1 if user_input.get('time_since_last_login', 24) < 0.1 else 0,
        'hour_deviation': hour_deviation,
        'user_login_frequency': user_input.get('user_login_frequency', 1),
        'weekday_deviation': abs(timestamp.weekday() - 2),
        'login_consistency': user_input.get('login_consistency', 0.7),
        'user_failure_rate': user_input.get('user_failure_rate', 0.1),
        'user_risk_score': user_input.get('user_risk_score', 0.3),
        'device_change': device_change,
        'user_device_diversity': user_input.get('user_device_diversity', 1),
        'country_change': country_change,
        'high_risk_country': high_risk_country,
        'user_country_diversity': user_input.get('user_country_diversity', 1),
        'login_failed': login_failed,
        'has_rtt': 1 if rtt is not None else 0,
        'rtt_vs_global': rtt_vs_global,
        'suspicious_combo': suspicious_combo,
        'failed_then_success': 0  # Cannot determine from single event
    }

def score_event(models, scaler, features):
    """Enhanced scoring with dynamic risk calculation"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config import RISK_THRESHOLDS
    
    feature_order = [
        "login_hour", "is_weekend", "is_night", "is_business_hours",
        "time_since_last_login", "rapid_login", "hour_deviation", "user_login_frequency",
        "weekday_deviation", "login_consistency", "user_failure_rate", "user_risk_score",
        "device_change", "user_device_diversity", "country_change", "high_risk_country", "user_country_diversity",
        "login_failed", "has_rtt", "rtt_vs_global", "suspicious_combo", "failed_then_success"
    ]
    
    ordered_features = [features.get(col, 0) for col in feature_order]
    feature_df = pd.DataFrame([ordered_features], columns=feature_order)
    X_scaled = scaler.transform(feature_df)
    
    # Enhanced ensemble scoring with better normalization
    scores = {}
    for name, model in models.items():
        if name == 'lof':
            raw_score = -model.decision_function(X_scaled)[0]
        else:
            raw_score = -model.decision_function(X_scaled)[0]
        scores[name] = raw_score
    
    # Improved score normalization
    normalized_scores = {}
    for name, score in scores.items():
        if name == 'isolation_forest':
            # Isolation Forest scores are typically between -1 and 1
            normalized_scores[name] = max(0, min(1, (score + 1) / 2))
        elif name == 'one_class_svm':
            # One-Class SVM scores need different normalization
            normalized_scores[name] = max(0, min(1, 1 / (1 + np.exp(-score))))
        else:
            # Default sigmoid normalization
            normalized_scores[name] = max(0, min(1, 1 / (1 + np.exp(-score))))
    
    # Dynamic weighted ensemble based on feature patterns
    base_weights = {'isolation_forest': 0.4, 'one_class_svm': 0.3, 'lof': 0.15, 'elliptic_envelope': 0.15}
    
    # Adjust weights based on anomaly patterns
    if features['suspicious_combo'] or features['device_change']:
        base_weights['isolation_forest'] += 0.2
        base_weights['one_class_svm'] += 0.1
    
    if features['login_failed'] or features['high_risk_country']:
        base_weights['lof'] += 0.1
        base_weights['elliptic_envelope'] += 0.1
    
    # Normalize weights
    total_weight = sum(base_weights.values())
    weights = {k: v/total_weight for k, v in base_weights.items()}
    
    base_score = sum(weights[name] * normalized_scores[name] for name in weights.keys())
    
    # Enhanced takeover boost with more nuanced scoring
    takeover_boost = 0
    boost_factors = []
    
    if features['suspicious_combo']:
        boost_factors.append(0.25)
    if features['failed_then_success']:
        boost_factors.append(0.3)
    if features['device_change'] and features['is_night']:
        boost_factors.append(0.15)
    if features['high_risk_country']:
        boost_factors.append(0.2)
    if features['rapid_login']:
        boost_factors.append(0.1)
    if features['hour_deviation'] > 6:  # Unusual time
        boost_factors.append(0.1)
    if features['rtt_vs_global'] > 300:  # High latency
        boost_factors.append(0.05)
    
    # Apply diminishing returns for multiple factors
    if boost_factors:
        takeover_boost = sum(boost_factors) * (0.8 ** (len(boost_factors) - 1))
    
    # Final score calculation with better balance
    final_score = np.clip(base_score * 0.7 + takeover_boost * 0.3, 0, 1)
    
    # Dynamic threshold adjustment based on patterns
    high_threshold = RISK_THRESHOLDS["high"]
    medium_threshold = RISK_THRESHOLDS["medium"]
    
    # Lower thresholds for high-risk patterns
    if features['suspicious_combo'] or features['high_risk_country']:
        high_threshold *= 0.9
        medium_threshold *= 0.9
    
    if final_score >= high_threshold:
        risk_level = "HIGH"
    elif final_score >= medium_threshold:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return final_score, risk_level

def show_real_time_scorer():
    """Enhanced real-time scoring interface with detailed analysis"""
    st.subheader("🔍 Real-time Event Scorer")
    
    models, scaler = load_models()
    if models is None:
        st.error("Models not found. Run training first.")
        return
    
    st.success(f"✅ Ensemble models loaded ({len(models)} models)")
    
    with st.form("scorer"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information**")
            user_id = st.text_input("User ID", "user123")
            login_date = st.date_input("Date", datetime.now().date())
            login_time = st.time_input("Time", datetime.now().time())
            success = st.checkbox("Login Successful", True)
        
        with col2:
            st.write("**Anomaly Indicators**")
            device_change = st.checkbox("New Device")
            country_change = st.checkbox("New Country")
            high_risk_country = st.checkbox("High-risk Country")
            rtt = st.number_input("Network Latency (ms)", 0, 2000, 200)
        
        with st.expander("Advanced User Profile Settings"):
            col3, col4 = st.columns(2)
            with col3:
                time_since_last = st.number_input("Hours Since Last Login", 0.0, 168.0, 24.0)
                typical_hour = st.number_input("Typical Login Hour", 0, 23, 12)
                login_frequency = st.number_input("Login Frequency/day", 0.1, 50.0, 2.0)
            with col4:
                user_failure_rate = st.slider("Historical Failure Rate", 0.0, 1.0, 0.1)
                login_consistency = st.slider("Login Consistency Score", 0.0, 1.0, 0.7)
                user_risk_score = st.slider("Baseline User Risk", 0.0, 1.0, 0.3)
        
        if st.form_submit_button("🎯 Score Event", type="primary"):
            user_input = {
                'user_id': user_id,
                'timestamp': datetime.combine(login_date, login_time),
                'success': success,
                'device_change': 1 if device_change else 0,
                'country_change': 1 if country_change else 0,
                'high_risk_country': 1 if high_risk_country else 0,
                'rtt': rtt,
                'time_since_last_login': time_since_last,
                'typical_hour': typical_hour,
                'user_login_frequency': login_frequency,
                'user_failure_rate': user_failure_rate,
                'login_consistency': login_consistency,
                'user_risk_score': user_risk_score
            }
            
            features = extract_features(user_input)
            risk_score, risk_level = score_event(models, scaler, features)
            
            # Enhanced Results Display
            st.divider()
            st.subheader("📈 Scoring Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{risk_score:.3f}")
            with col2:
                color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[risk_level]
                st.metric("Risk Level", f"{color} {risk_level}")
            with col3:
                st.metric("User", user_id)
            
            # Detailed Risk Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Risk Factors Detected:**")
                risk_factors = []
                risk_scores = []
                
                if features['suspicious_combo']:
                    risk_factors.append("Device + Country change combo")
                    risk_scores.append(0.25)
                if features['device_change']:
                    risk_factors.append("New device detected")
                    risk_scores.append(0.15)
                if features['country_change']:
                    risk_factors.append("New country detected")
                    risk_scores.append(0.15)
                if features['high_risk_country']:
                    risk_factors.append("High-risk country")
                    risk_scores.append(0.20)
                if features['login_failed']:
                    risk_factors.append("Failed login attempt")
                    risk_scores.append(0.15)
                if features['is_night']:
                    risk_factors.append("Night-time login")
                    risk_scores.append(0.10)
                if features['rapid_login']:
                    risk_factors.append("Rapid successive login")
                    risk_scores.append(0.10)
                if features['hour_deviation'] > 6:
                    risk_factors.append(f"Unusual time (deviation: {features['hour_deviation']:.1f}h)")
                    risk_scores.append(0.08)
                if features['rtt_vs_global'] > 300:
                    risk_factors.append(f"High network latency ({rtt}ms)")
                    risk_scores.append(0.05)
                
                if risk_factors:
                    for factor, score in zip(risk_factors, risk_scores):
                        st.write(f"• {factor} (+{score:.2f})")
                else:
                    st.write("✅ No significant risk factors detected")
            
            with col2:
                st.write("**Feature Analysis:**")
                
                # Show key feature values
                key_features = {
                    'Time Deviation': f"{features['hour_deviation']:.1f} hours",
                    'Login Frequency': f"{features['user_login_frequency']:.1f}/day",
                    'Failure Rate': f"{features['user_failure_rate']:.1%}",
                    'Consistency Score': f"{features['login_consistency']:.2f}",
                    'Network Latency': f"{rtt}ms",
                    'Time Since Last': f"{time_since_last:.1f}h"
                }
                
                for feature, value in key_features.items():
                    st.write(f"• **{feature}**: {value}")
            
            # Risk Score Breakdown
            st.write("**Risk Score Breakdown:**")
            
            # Calculate component scores (simplified for display)
            base_anomaly = min(0.6, risk_score * 0.7)
            takeover_boost = max(0, risk_score - base_anomaly)
            
            progress_col1, progress_col2 = st.columns(2)
            
            with progress_col1:
                st.write(f"Base Anomaly Score: {base_anomaly:.3f}")
                st.progress(base_anomaly)
            
            with progress_col2:
                st.write(f"Takeover Boost: {takeover_boost:.3f}")
                st.progress(min(1.0, takeover_boost * 2))  # Scale for visibility
            
            # Recommendation
            if risk_level == "HIGH":
                st.error("⚠️ **HIGH RISK**: Immediate investigation recommended. Potential account takeover.")
            elif risk_level == "MEDIUM":
                st.warning("🔍 **MEDIUM RISK**: Monitor closely. Additional verification may be needed.")
            else:
                st.success("✅ **LOW RISK**: Normal behavior pattern detected.")
            
            # Show threshold context
            st.write("**Threshold Context:**")
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from config import RISK_THRESHOLDS
            
            threshold_col1, threshold_col2, threshold_col3 = st.columns(3)
            with threshold_col1:
                st.write(f"Low: < {RISK_THRESHOLDS['medium']:.2f}")
            with threshold_col2:
                st.write(f"Medium: {RISK_THRESHOLDS['medium']:.2f} - {RISK_THRESHOLDS['high']:.2f}")
            with threshold_col3:
                st.write(f"High: > {RISK_THRESHOLDS['high']:.2f}")