import pandas as pd
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import pickle

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_THRESHOLDS, FEATURE_WEIGHTS, NETWORK_PARAMS
from src.ueba_analytics import UserBehavioralAnalytics

# Model configuration with higher contamination for better detection
MODEL_CONFIG = {
    "algorithm": "IsolationForest",
    "n_estimators": 200,
    "contamination": 0.02,  # Increased from 0.01 to catch more anomalies
    "random_state": 42
}

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "preprocessed_logs.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "alerts_ready.csv")

# Enhanced feature set with takeover detection
FEATURE_COLS = [
    "login_hour", "is_weekend", "is_night", "is_business_hours",
    "time_since_last_login", "rapid_login", "hour_deviation", "user_login_frequency",
    "weekday_deviation", "login_consistency", "user_failure_rate", "user_risk_score",
    "device_change", "user_device_diversity", "country_change", "high_risk_country", "user_country_diversity",
    "login_failed", "has_rtt", "rtt_vs_global", "suspicious_combo", "failed_then_success"
]

def explain_anomaly(row):
    """Generate weighted explanation for why a login was flagged"""
    reasons = []
    
    if row.get("suspicious_combo", 0) == 1:
        reasons.append("Device + Country change combo")
    if row.get("failed_then_success", 0) == 1:
        reasons.append("Failed login then success pattern")
    if row.get("device_change", 0) == 1:
        reasons.append("New device used")
    if row.get("country_change", 0) == 1:
        reasons.append("New country detected")
    if row.get("high_risk_country", 0) == 1:
        reasons.append("High-risk country")
    if row.get("login_failed", 0) == 1:
        reasons.append("Failed login attempt")
    if row.get("is_night", 0) == 1:
        reasons.append("Night-time login")
    if row.get("rapid_login", 0) == 1:
        reasons.append("Rapid successive login")
    if row.get("hour_deviation", 0) > 6:
        reasons.append("Unusual login time")
    if abs(row.get("rtt_vs_global", 0)) > NETWORK_PARAMS["high_latency_threshold"]:
        reasons.append("Unusual network latency")
    
    return "; ".join(reasons) if reasons else "Behavior deviates from baseline"

def assign_risk_level(score):
    """Convert continuous risk score to categorical alert level using config thresholds"""
    if score >= RISK_THRESHOLDS["high"]:
        return "HIGH"
    elif score >= RISK_THRESHOLDS["medium"]:
        return "MEDIUM"
    else:
        return "LOW"

def train_ensemble_models(X):
    """Train multiple anomaly detection models"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {}
    
    # Isolation Forest
    models['isolation_forest'] = IsolationForest(
        n_estimators=MODEL_CONFIG["n_estimators"],
        contamination=MODEL_CONFIG["contamination"],
        random_state=MODEL_CONFIG["random_state"]
    )
    
    # One-Class SVM
    models['one_class_svm'] = OneClassSVM(
        gamma='scale',
        nu=MODEL_CONFIG["contamination"]
    )
    
    # Local Outlier Factor
    models['lof'] = LocalOutlierFactor(
        n_neighbors=20,
        contamination=MODEL_CONFIG["contamination"],
        novelty=True
    )
    
    # Elliptic Envelope
    models['elliptic_envelope'] = EllipticEnvelope(
        contamination=MODEL_CONFIG["contamination"],
        random_state=MODEL_CONFIG["random_state"]
    )
    
    # Train all models
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_scaled)
    
    return models, scaler

def calculate_takeover_boost(df):
    """Calculate takeover-specific boost scores - enhanced for better separation"""
    boost_scores = np.zeros(len(df))
    
    # Strong takeover pattern: device + country change combo
    suspicious_combo_boost = df['suspicious_combo'].fillna(0) * 0.4
    
    # Failed then success pattern (common in takeovers)
    failed_success_boost = df['failed_then_success'].fillna(0) * 0.35
    
    # Compound boost for multiple strong indicators
    both_patterns = (
        (df['suspicious_combo'].fillna(0) == 1) & 
        (df['failed_then_success'].fillna(0) == 1)
    )
    
    # Base boosts
    boost_scores += suspicious_combo_boost
    boost_scores += failed_success_boost
    
    # Additional boost for events with both strong patterns
    boost_scores[both_patterns] += 0.25  # Extra boost for clearest takeovers
    
    # Night + device change (suspicious timing)
    night_device_boost = (df['device_change'].fillna(0) & df['is_night'].fillna(0)) * 0.15
    boost_scores += night_device_boost
    
    return np.clip(boost_scores, 0, 1)

def generate_ensemble_scores(models, scaler, X, df):
    """Generate ensemble anomaly scores with takeover-specific weighting"""
    X_scaled = scaler.transform(X)
    
    # Get scores from each model
    scores = {}
    for name, model in models.items():
        if name == 'lof':
            scores[name] = -model.decision_function(X_scaled)
        else:
            scores[name] = -model.decision_function(X_scaled)
    
    # Normalize each model's scores with NaN handling
    normalized_scores = {}
    for name, score_array in scores.items():
        # Handle NaN values
        valid_scores = score_array[~np.isnan(score_array)]
        if len(valid_scores) == 0:
            normalized_scores[name] = np.zeros(len(score_array))
            continue
            
        p95 = np.percentile(valid_scores, 95)
        p5 = np.percentile(valid_scores, 5)
        
        if p95 == p5:  # All values are the same
            normalized_scores[name] = np.zeros(len(score_array))
        else:
            norm_scores = (score_array - p5) / (p95 - p5)
            norm_scores = np.nan_to_num(norm_scores, nan=0.0)
            normalized_scores[name] = np.clip(norm_scores, 0, 1)
    
    # Weighted ensemble favoring Isolation Forest (best performer)
    weights = {'isolation_forest': 0.7, 'one_class_svm': 0.2, 'lof': 0.05, 'elliptic_envelope': 0.05}
    
    ensemble_scores = np.zeros(len(X))
    for name, weight in weights.items():
        ensemble_scores += weight * normalized_scores[name]
    
    # Calculate takeover-specific boost
    takeover_boost = calculate_takeover_boost(df)
    
    # Combine base anomaly score with takeover boost with NaN handling
    final_scores = ensemble_scores * 0.6 + takeover_boost * 0.4
    final_scores = np.nan_to_num(final_scores, nan=0.0)
    
    # Apply power transformation
    final_scores = np.power(np.clip(final_scores, 0, 1), 0.7)
    
    return final_scores, normalized_scores

def print_model_summary(df, individual_scores):
    """Print summary statistics and validation results"""
    print("\n=== Ensemble Model Results Summary ===")
    print(f"Total events processed: {len(df)}")
    print(f"Risk score range: {df['risk_score'].min():.3f} - {df['risk_score'].max():.3f}")
    
    print("\nRisk level distribution:")
    risk_dist = df["risk_level"].value_counts(normalize=True)
    for level, pct in risk_dist.items():
        print(f"  {level}: {pct:.1%}")
    
    # Model performance comparison
    if "Is Account Takeover" in df.columns:
        print("\nModel Performance Comparison:")
        takeover_mask = df["Is Account Takeover"] == True
        
        print(f"Ensemble Score - Takeovers: {df.loc[takeover_mask, 'risk_score'].mean():.3f}")
        print(f"Ensemble Score - Normal: {df.loc[~takeover_mask, 'risk_score'].mean():.3f}")
        
        for model_name in individual_scores.keys():
            takeover_avg = individual_scores[model_name][takeover_mask].mean()
            normal_avg = individual_scores[model_name][~takeover_mask].mean()
            print(f"{model_name} - Takeovers: {takeover_avg:.3f}, Normal: {normal_avg:.3f}")
    
    print("\nTop 5 highest risk events:")
    top_risks = df.nlargest(5, "risk_score")[
        ["User ID", "risk_score", "risk_level", "explanation"]
    ]
    print(top_risks.to_string(index=False))

def main():
    """Main ensemble model training and scoring pipeline with UEBA analytics"""
    print("Loading preprocessed data...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Loaded {len(df)} login events with {len(FEATURE_COLS)} features")
    
    # Check if all features exist
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        print(f"Warning: Missing features {missing_features}. Using available features only.")
        available_features = [col for col in FEATURE_COLS if col in df.columns]
        X = df[available_features]
    else:
        X = df[FEATURE_COLS]
    
    print("Training ensemble models...")
    models, scaler = train_ensemble_models(X)
    
    print("Generating ensemble risk scores...")
    ensemble_scores, individual_scores = generate_ensemble_scores(models, scaler, X, df)
    
    # Add results to dataframe
    df["risk_score"] = ensemble_scores
    df["risk_level"] = df["risk_score"].apply(assign_risk_level)
    df["explanation"] = df.apply(explain_anomaly, axis=1)
    
    # Add individual model scores for comparison
    for model_name, scores in individual_scores.items():
        df[f"{model_name}_score"] = scores
    
    # UEBA Analytics
    print("\nGenerating UEBA analytics...")
    ueba = UserBehavioralAnalytics()
    
    # Build user baselines
    print("Building user behavioral baselines...")
    user_baselines = ueba.build_user_baselines(df)
    
    # Compute behavioral deviations
    print("Computing behavioral deviations...")
    deviations_df = ueba.compute_behavioral_deviations(df, user_baselines)
    
    # Add deviation scores to main dataframe
    for col in deviations_df.columns:
        df[col] = deviations_df[col]
    
    # Track risk progressions
    print("Tracking risk progressions...")
    risk_progressions = ueba.track_risk_progression(df)
    
    # Print summary
    print_model_summary(df, individual_scores)
    
    # UEBA Summary
    print("\n=== UEBA Analytics Summary ===")
    print(f"User baselines created: {len(user_baselines)}")
    print(f"Users with risk progressions: {len(risk_progressions)}")
    
    # High-risk users summary
    high_risk_users = [uid for uid, prog in risk_progressions.items() 
                      if prog.get('current_trend', 0) > 0.1 and prog.get('max_risk_score', 0) > 0.7]
    print(f"High-risk trending users: {len(high_risk_users)}")
    
    if high_risk_users:
        print("Top concerning users:")
        for user_id in high_risk_users[:5]:
            summary = ueba.generate_user_risk_summary(user_id, df, user_baselines, risk_progressions)
            if summary:
                print(f"  {user_id}: {summary['recommendation']}")
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")
    
    # Save ensemble models and UEBA data
    model_path = os.path.join(BASE_DIR, "..", "ensemble_models.pkl")
    scaler_path = os.path.join(BASE_DIR, "..", "scaler.pkl")
    ueba_path = os.path.join(BASE_DIR, "..", "ueba_data.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(ueba_path, 'wb') as f:
        pickle.dump({
            'user_baselines': user_baselines,
            'risk_progressions': risk_progressions,
            'ueba_analytics': ueba
        }, f)
    
    print(f"Ensemble models saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"UEBA data saved to: {ueba_path}")
    
    print("\nModel configuration:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()