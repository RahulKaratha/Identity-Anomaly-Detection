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
    return df

def create_change_features(df):
    """Detect changes in user behavior patterns"""
    # Device change detection
    df["prev_device"] = df.groupby("User ID")["Device Type"].shift(1)
    df["device_change"] = (df["Device Type"] != df["prev_device"]).astype(int)
    df["device_change"] = df["device_change"].fillna(0)
    
    # Country change detection
    df["prev_country"] = df.groupby("User ID")["Country"].shift(1)
    df["country_change"] = (df["Country"] != df["prev_country"]).astype(int)
    df["country_change"] = df["country_change"].fillna(0)
    
    # Clean up temporary columns
    df = df.drop(["prev_device", "prev_country"], axis=1)
    
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

def main():
    """Main preprocessing pipeline"""
    print("Loading raw login data...")
    df = load_and_clean_data()
    
    print("Creating behavioral features...")
    df = create_time_features(df)
    df = create_change_features(df)
    df = create_failure_features(df)
    df = create_network_features(df)
    
    # Define final feature set
    feature_cols = [
        "login_hour", "is_weekend", "device_change", 
        "country_change", "login_failed", "has_rtt", "rtt_vs_global"
    ]
    
    # Data quality checks
    print("\nData quality summary:")
    print(f"Total records: {len(df)}")
    print(f"Missing values in features: {df[feature_cols].isna().sum().sum()}")
    print(f"Unique users: {df['User ID'].nunique()}")
    
    # Save processed data
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPreprocessed data saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
