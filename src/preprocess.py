import pandas as pd
import os

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "login_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "preprocessed_logs.csv")

def load_and_clean_data():
    """Load raw login data and handle basic cleaning"""
    df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)
    
    # Parse timestamps
    df["Login Timestamp"] = pd.to_datetime(df["Login Timestamp"], errors="coerce")
    
    # Sort by user and time for sequential features
    df = df.sort_values(by=["User ID", "Login Timestamp"])
    
    return df

def create_time_features(df):
    """Extract time-based behavioral features"""
    df["login_hour"] = df["Login Timestamp"].dt.hour
    df["login_day"] = df["Login Timestamp"].dt.dayofweek
    df["is_weekend"] = df["login_day"].isin([5, 6]).astype(int)
    
    # Enhanced time features
    df["is_night"] = df["login_hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df["is_business_hours"] = df["login_hour"].isin(range(9, 18)).astype(int)
    
    return df

def create_session_features(df):
    """Create session-based behavioral features"""
    # Calculate time between consecutive logins for each user
    df["prev_login"] = df.groupby("User ID")["Login Timestamp"].shift(1)
    df["time_since_last_login"] = (df["Login Timestamp"] - df["prev_login"]).dt.total_seconds() / 3600  # hours
    df["time_since_last_login"] = df["time_since_last_login"].fillna(24)  # Default to 24h for first login
    
    # Rapid successive logins (potential bot behavior)
    df["rapid_login"] = (df["time_since_last_login"] < 0.1).astype(int)  # Less than 6 minutes
    
    # Clean up temporary column
    df = df.drop(["prev_login"], axis=1)
    
    return df

def create_user_behavior_features(df):
    """Create user-specific behavioral patterns"""
    # Calculate user's typical login hour
    user_typical_hour = df.groupby("User ID")["login_hour"].transform("median")
    df["hour_deviation"] = abs(df["login_hour"] - user_typical_hour)
    
    # User login frequency (logins per day)
    user_login_counts = df.groupby("User ID").size()
    date_range = (df["Login Timestamp"].max() - df["Login Timestamp"].min()).days + 1
    user_daily_frequency = user_login_counts / date_range
    df["user_login_frequency"] = df["User ID"].map(user_daily_frequency)
    
    return df

def create_advanced_user_profiles(df):
    """Advanced user behavior profiling"""
    # Weekly patterns
    user_weekday_pref = df.groupby("User ID")["login_day"].apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0)
    df["weekday_deviation"] = abs(df["login_day"] - df["User ID"].map(user_weekday_pref))
    
    # Login consistency score (lower = more consistent)
    user_hour_std = df.groupby("User ID")["login_hour"].transform("std").fillna(0)
    df["login_consistency"] = 1 / (1 + user_hour_std)  # Higher = more consistent
    
    # Failure rate per user
    user_failure_rate = df.groupby("User ID")["login_failed"].transform("mean")
    df["user_failure_rate"] = user_failure_rate
    
    # User risk score (based on historical behavior)
    df["user_risk_score"] = (
        df["user_failure_rate"] * 0.4 + 
        (1 - df["login_consistency"]) * 0.3 + 
        (df["user_country_diversity"] > 2).astype(int) * 0.3
    )
    
    return df

def create_geographic_features(df):
    """Enhanced geographic risk assessment"""
    # High-risk countries (simplified list)
    high_risk_countries = ['Unknown', 'Anonymous', 'TOR']
    df["high_risk_country"] = df["Country"].isin(high_risk_countries).astype(int)
    
    # Multiple countries in short time (potential VPN/proxy use)
    df["prev_country"] = df.groupby("User ID")["Country"].shift(1)
    df["country_change"] = (df["Country"] != df["prev_country"]).astype(int)
    df["country_change"] = df["country_change"].fillna(0)
    
    # Count unique countries per user
    user_country_counts = df.groupby("User ID")["Country"].nunique()
    df["user_country_diversity"] = df["User ID"].map(user_country_counts)
    
    # Clean up temporary column
    df = df.drop(["prev_country"], axis=1)
    
    return df

def create_device_features(df):
    """Enhanced device-based features"""
    # Device change detection
    df["prev_device"] = df.groupby("User ID")["Device Type"].shift(1)
    df["device_change"] = (df["Device Type"] != df["prev_device"]).astype(int)
    df["device_change"] = df["device_change"].fillna(0)
    
    # Count unique devices per user
    user_device_counts = df.groupby("User ID")["Device Type"].nunique()
    df["user_device_diversity"] = df["User ID"].map(user_device_counts)
    
    # Clean up temporary column
    df = df.drop(["prev_device"], axis=1)
    
    return df

def create_failure_features(df):
    """Create features related to login failures"""
    df["login_failed"] = (~df["Login Successful"]).astype(int)
    return df

def create_network_features(df):
    """Create network latency features"""
    df["Round-Trip Time [ms]"] = pd.to_numeric(df["Round-Trip Time [ms]"], errors="coerce")
    df["has_rtt"] = df["Round-Trip Time [ms]"].notna().astype(int)
    
    # Compare to global average
    global_rtt_mean = df["Round-Trip Time [ms]"].mean()
    df["rtt_vs_global"] = df["Round-Trip Time [ms]"] - global_rtt_mean
    df["rtt_vs_global"] = df["rtt_vs_global"].fillna(0)
    
    return df

def create_takeover_detection_features(df):
    """Create features specifically designed to detect account takeovers"""
    # Multiple suspicious activities in short time
    df['suspicious_combo'] = (
        (df['device_change'] == 1) & 
        (df['country_change'] == 1)
    ).astype(int)
    
    # Failed login followed by successful login (common takeover pattern)
    df['prev_failed'] = df.groupby('User ID')['login_failed'].shift(1).fillna(0)
    df['failed_then_success'] = (
        (df['prev_failed'] == 1) & 
        (df['login_failed'] == 0)
    ).astype(int)
    
    # Clean up temporary columns
    df = df.drop(['prev_failed'], axis=1)
    
    return df

def main():
    """Main preprocessing pipeline"""
    print("Loading raw login data...")
    df = load_and_clean_data()
    
    print("Creating behavioral features...")
    df = create_time_features(df)
    df = create_session_features(df)
    df = create_user_behavior_features(df)
    df = create_device_features(df)
    df = create_geographic_features(df)
    df = create_failure_features(df)
    df = create_network_features(df)
    df = create_takeover_detection_features(df)
    df = create_advanced_user_profiles(df)  # Move after failure features
    
    # Enhanced feature set with advanced profiling
    feature_cols = [
        "login_hour", "is_weekend", "is_night", "is_business_hours",
        "time_since_last_login", "rapid_login", "hour_deviation", "user_login_frequency",
        "weekday_deviation", "login_consistency", "user_failure_rate", "user_risk_score",
        "device_change", "user_device_diversity", "country_change", "high_risk_country", "user_country_diversity",
        "login_failed", "has_rtt", "rtt_vs_global", "suspicious_combo", "failed_then_success"
    ]
    
    # Data quality checks
    print("\nData quality summary:")
    print(f"Total records: {len(df)}")
    print(f"Missing values in features: {df[feature_cols].isna().sum().sum()}")
    print(f"Unique users: {df['User ID'].nunique()}")
    print(f"Enhanced features: {len(feature_cols)}")
    
    # SavEe processed data
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPreprocessed data saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
