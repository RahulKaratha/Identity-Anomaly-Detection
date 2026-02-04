# Configuration file for Identity Anomaly Detection

# Model parameters
MODEL_CONFIG = {
    "algorithm": "IsolationForest",
    "n_estimators": 200,
    "contamination": 0.01,  # Expected percentage of anomalies
    "random_state": 42
}

# Risk scoring thresholds - optimized for quality alerts
RISK_THRESHOLDS = {
    "high": 0.75,   # Maintain current HIGH threshold
    "medium": 0.6,  # Raised to reduce MEDIUM alerts
    "low": 0.0
}

# Feature weights for explanation scoring
FEATURE_WEIGHTS = {
    "device_change": 0.8,
    "country_change": 0.9,
    "login_failed": 0.7,
    "is_night": 0.6,
    "rapid_login": 0.8,
    "high_risk_country": 0.9,
    "hour_deviation": 0.5
}

# Time-based parameters
TIME_PARAMS = {
    "rapid_login_threshold": 0.1,  # hours
    "night_hours": [22, 23, 0, 1, 2, 3, 4, 5],
    "business_hours": list(range(9, 18))
}

# Geographic risk assessment
HIGH_RISK_COUNTRIES = [
    'Unknown', 'Anonymous', 'TOR', 'Proxy'
]

# Network parameters
NETWORK_PARAMS = {
    "high_latency_threshold": 500  # milliseconds
}