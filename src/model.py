import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "preprocessed_logs.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "alerts_ready.csv")

# Model configuration
MODEL_CONFIG = {
    "algorithm": "IsolationForest",
    "n_estimators": 200,
    "contamination": 0.01,  # Expect 1% of logins to be anomalous
    "random_state": 42
}

# Features used by the model
FEATURE_COLS = [
    "login_hour", "is_weekend", "device_change", 
    "country_change", "login_failed", "has_rtt", "rtt_vs_global"
]

def explain_anomaly(row):
    """Generate human-readable explanation for why a login was flagged"""
    reasons = []
    
    if row["device_change"] == 1:
        reasons.append("New device used")
    if row["country_change"] == 1:
        reasons.append("New country detected")
    if row["login_failed"] == 1:
        reasons.append("Failed login attempt")
    if row["is_weekend"] == 1:
        reasons.append("Weekend login")
    if abs(row["rtt_vs_global"]) > 500:
        reasons.append("Unusual network latency")
    
    return "; ".join(reasons) if reasons else "Behavior deviates from baseline"

def assign_risk_level(score):
    """Convert continuous risk score to categorical alert level"""
    if score >= 0.7:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

def train_isolation_forest(X):
    """Train and return Isolation Forest model"""
    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = IsolationForest(
        n_estimators=MODEL_CONFIG["n_estimators"],
        contamination=MODEL_CONFIG["contamination"],
        random_state=MODEL_CONFIG["random_state"]
    )
    model.fit(X_scaled)
    
    return model, scaler

def generate_risk_scores(model, scaler, X):
    """Generate normalized risk scores from model predictions"""
    X_scaled = scaler.transform(X)
    
    # Get anomaly scores (more negative = more anomalous)
    anomaly_scores = -model.decision_function(X_scaled)
    
    # Normalize to 0-1 range
    min_score, max_score = anomaly_scores.min(), anomaly_scores.max()
    risk_scores = (anomaly_scores - min_score) / (max_score - min_score)
    
    return anomaly_scores, risk_scores

def print_model_summary(df):
    """Print summary statistics and validation results"""
    print("\n=== Model Results Summary ===")
    print(f"Total events processed: {len(df)}")
    print(f"Risk score range: {df['risk_score'].min():.3f} - {df['risk_score'].max():.3f}")
    
    print("\nRisk level distribution:")
    risk_dist = df["risk_level"].value_counts(normalize=True)
    for level, pct in risk_dist.items():
        print(f"  {level}: {pct:.1%}")
    
    # Validation against known takeovers if available
    if "Is Account Takeover" in df.columns:
        print("\nValidation against known account takeovers:")
        takeover_stats = df.groupby("Is Account Takeover")["risk_score"].agg(['mean', 'std'])
        print(takeover_stats)
    
    print("\nTop 5 highest risk events:")
    top_risks = df.nlargest(5, "risk_score")[
        ["User ID", "risk_score", "risk_level", "explanation"]
    ]
    print(top_risks.to_string(index=False))

def main():
    """Main model training and scoring pipeline"""
    print("Loading preprocessed data...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Loaded {len(df)} login events with {len(FEATURE_COLS)} features")
    
    # Prepare feature matrix
    X = df[FEATURE_COLS]
    
    print("Training Isolation Forest model...")
    model, scaler = train_isolation_forest(X)
    
    print("Generating risk scores...")
    anomaly_scores, risk_scores = generate_risk_scores(model, scaler, X)
    
    # Add results to dataframe
    df["anomaly_score"] = anomaly_scores
    df["risk_score"] = risk_scores
    df["risk_level"] = df["risk_score"].apply(assign_risk_level)
    df["explanation"] = df.apply(explain_anomaly, axis=1)
    
    # Print summary
    print_model_summary(df)
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")
    
    print("\nModel configuration:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
