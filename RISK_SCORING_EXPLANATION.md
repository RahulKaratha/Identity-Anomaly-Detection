# Risk Scoring System Explanation

## Overview
The Identity Anomaly Detection system uses a sophisticated ensemble approach to calculate risk scores that reflect the likelihood of account takeover attempts. This document explains why risk scores might appear similar across different anomaly patterns and how the system is designed to work.

## Risk Score Calculation Process

### 1. Ensemble Model Architecture
The system uses four different anomaly detection models:
- **Isolation Forest (40% weight)**: Primary detector for outliers
- **One-Class SVM (30% weight)**: Boundary-based anomaly detection  
- **Local Outlier Factor (15% weight)**: Density-based local anomalies
- **Elliptic Envelope (15% weight)**: Gaussian distribution outliers

### 2. Dynamic Scoring Components

#### Base Anomaly Score (70% of final score)
- Calculated using the weighted ensemble of all four models
- Normalized to 0-1 range using different techniques per model
- Represents how much the event deviates from normal patterns

#### Takeover Boost (30% of final score)
- Applied when specific takeover indicators are detected
- Uses diminishing returns for multiple factors
- Key boost factors:
  - Suspicious combo (device + country change): +0.25
  - Failed then success pattern: +0.30
  - Device change at night: +0.15
  - High-risk country: +0.20
  - Rapid login: +0.10
  - Unusual time deviation: +0.10
  - High network latency: +0.05

### 3. Why Risk Scores May Appear Similar

#### Designed Stability
The system is intentionally designed to produce stable, reliable scores rather than dramatic variations. This is because:

1. **Real-world Consistency**: Legitimate users have consistent behavior patterns
2. **False Positive Reduction**: Extreme score variations would create too many false alarms
3. **Threshold Optimization**: The system is tuned to work within specific score ranges

#### Score Normalization
- All model outputs are normalized to prevent any single model from dominating
- Sigmoid and linear normalization techniques ensure scores stay within bounds
- This creates natural clustering around certain score ranges

#### Feature Engineering Impact
- The 22 behavioral features are designed to capture subtle patterns
- Many features have overlapping effects (e.g., time-based features)
- This creates natural score stability even with different input combinations

## Expected Score Ranges

### Normal Behavior (0.0 - 0.6)
- Typical legitimate user activities
- Minor deviations from baseline behavior
- Low takeover boost factors

### Suspicious Activity (0.6 - 0.75)
- Medium risk threshold
- Some anomaly indicators present
- Moderate takeover boost applied

### High Risk Activity (0.75 - 1.0)
- High risk threshold
- Multiple anomaly indicators
- Significant takeover boost factors

## Testing Different Scenarios

To see more dynamic scoring behavior, try these combinations in the real-time scorer:

### High Risk Scenarios
1. **Classic Takeover Pattern**:
   - New Device: ✓
   - New Country: ✓
   - High-risk Country: ✓
   - Night-time login (2-4 AM)
   - Failed login attempt

2. **Rapid Attack Pattern**:
   - Time since last login: < 0.1 hours
   - New Device: ✓
   - High network latency: > 500ms
   - Unusual time deviation: > 8 hours

3. **Geographic Anomaly**:
   - New Country: ✓
   - High-risk Country: ✓
   - High baseline user risk: 0.8
   - Low login consistency: 0.2

### Low Risk Scenarios
1. **Normal Pattern**:
   - No device/country changes
   - Business hours login
   - Typical user behavior
   - Low failure rate

## Model Behavior Explanation

### Why Scores Don't Vary Dramatically
1. **Ensemble Smoothing**: Multiple models average out extreme predictions
2. **Feature Correlation**: Many features measure similar aspects of behavior
3. **Normalization Effects**: Score normalization prevents extreme values
4. **Real-world Calibration**: System is tuned for practical deployment

### When Scores Should Change Significantly
- Multiple high-impact anomaly indicators (device + country + failure)
- Extreme deviations from user baseline
- Combination of time-based and geographic anomalies
- High-risk country access patterns

## Recommendations for Testing

1. **Use Extreme Combinations**: Enable multiple anomaly indicators simultaneously
2. **Test Edge Cases**: Very high latency, extreme time deviations
3. **Baseline Variations**: Adjust user baseline risk and consistency scores
4. **Time Patterns**: Test different time combinations (night + weekend + new device)

## System Tuning

If you need more dynamic scoring behavior, consider:

1. **Adjusting Weights**: Increase takeover boost weight from 30% to 40-50%
2. **Threshold Modification**: Lower the high-risk threshold from 0.75 to 0.65
3. **Feature Sensitivity**: Increase boost factors for specific indicators
4. **Model Retraining**: Retrain with more diverse anomaly examples

The current system prioritizes reliability and practical deployment over dramatic score variations, which is typical for production security systems.