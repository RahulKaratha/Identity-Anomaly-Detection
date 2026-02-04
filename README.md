# Identity Anomaly Detection - Advanced UEBA System

An advanced User and Entity Behavior Analytics (UEBA) system that detects account takeovers and suspicious login behavior using ensemble machine learning models with takeover-specific detection capabilities.

## What it does

This enterprise-grade security tool monitors user login activities and identifies account takeovers and security threats through advanced behavioral analysis. It specializes in detecting:

- **Account takeovers**: Device + location changes with failed-then-success patterns
- **Suspicious login patterns**: Night-time access, rapid logins, geographic anomalies
- **Behavioral deviations**: Changes from established user baselines
- **Risk progression**: Escalating threat patterns over time
- **Advanced persistent threats**: Sophisticated attack patterns

## System Architecture

### Enhanced Detection Engine
- **Ensemble Models**: Isolation Forest (70%), One-Class SVM (20%), LOF (5%), Elliptic Envelope (5%)
- **Takeover-Specific Scoring**: Specialized boost algorithms for account takeover detection
- **UEBA Analytics**: Personalized user baselines and behavioral deviation tracking
- **22 Behavioral Features**: Advanced feature engineering for comprehensive analysis

### Performance Metrics
- **Takeover Detection Score**: 0.676 (vs 0.507 for normal events)
- **Alert Distribution**: 2.9% HIGH, 17.4% MEDIUM, 79.7% LOW risk
- **Processing Capacity**: 400,000+ events with real-time scoring
- **Detection Accuracy**: Optimized for minimal false positives

## How to use

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data**
   - Place your login data in `data/login_data.csv`
   - Required columns: User ID, Login Timestamp, Device Type, Country, Login Successful, Round-Trip Time [ms], IP Address, Is Account Takeover

3. **Run the analysis**
   ```bash
   # Process the raw data with enhanced features
   python src/preprocess.py
   
   # Train ensemble models and generate alerts
   python src/model.py
   ```

4. **Launch the UEBA Dashboard**
   ```bash
   streamlit run dashboard/dashboard.py
   ```
   
   **Dashboard Features:**
   - **High-Risk Users Overview**: Risk timeline visualizations with hover tooltips
   - **Individual User Deep-Dive**: Comprehensive behavioral analysis
   - **Real-time Event Scorer**: Instant risk assessment for new events
   - **Interactive Visualizations**: Plotly-powered charts with detailed explanations

## Enhanced Project Structure

```
├── src/
│   ├── preprocess.py           # 22-feature behavioral analysis
│   ├── model.py               # Ensemble models with takeover detection
│   ├── ueba_analytics.py      # User behavioral analytics engine
│   └── ueba_visualizations.py # Interactive Plotly visualizations
├── dashboard/
│   └── dashboard.py           # Multi-page UEBA dashboard
├── data/
│   ├── login_data.csv         # Raw login events (400k+ records)
│   ├── preprocessed_logs.csv  # Enhanced behavioral features
│   └── alerts_ready.csv       # Risk scores with UEBA analytics
├── config.py                  # Centralized configuration
├── ensemble_models.pkl        # Trained ensemble models
├── scaler.pkl                # Feature scaling parameters
├── ueba_data.pkl             # User baselines and risk progressions
└── requirements.txt
```

## Advanced Detection Capabilities

### Takeover-Specific Features
- **suspicious_combo**: Simultaneous device + country changes
- **failed_then_success**: Failed login followed by successful access
- **Geographic intelligence**: High-risk country detection and travel patterns
- **Temporal analysis**: Night-time access and unusual timing
- **Session patterns**: Rapid logins and behavioral consistency

### UEBA Analytics Engine
- **Personalized Baselines**: Individual user behavior profiles
- **Risk Progression Tracking**: Escalating threat detection
- **Behavioral Deviation Scoring**: Quantified changes from normal patterns
- **Trend Analysis**: Risk trajectory monitoring

### Enhanced Behavioral Features (22 Total)
1. **Time Analysis**: login_hour, is_weekend, is_night, is_business_hours
2. **Session Patterns**: time_since_last_login, rapid_login, hour_deviation
3. **User Profiling**: user_login_frequency, weekday_deviation, login_consistency
4. **Risk Assessment**: user_failure_rate, user_risk_score
5. **Device Intelligence**: device_change, user_device_diversity
6. **Geographic Analysis**: country_change, high_risk_country, user_country_diversity
7. **Security Events**: login_failed, has_rtt, rtt_vs_global
8. **Takeover Detection**: suspicious_combo, failed_then_success

## Configuration Management

**Risk Thresholds** (Optimized for Quality):
- HIGH: 0.75 (Targets ~1000-2000 critical alerts)
- MEDIUM: 0.6 (Investigation-worthy events)
- LOW: 0.0 (Baseline monitoring)

**Model Configuration**:
- Ensemble approach with weighted scoring
- Contamination rate: 0.02 (2% expected anomalies)
- Takeover boost scoring for enhanced detection

## Dashboard Capabilities

### High-Risk Users Overview
- **Risk Timeline**: Interactive time-series with hover details
- **User Risk Profiles**: Comprehensive behavioral summaries
- **Trend Visualization**: Risk progression over time
- **Alert Prioritization**: Focus on highest-risk users

### Individual User Deep-Dive
- **Behavioral Baselines**: Personal normal patterns
- **Deviation Analysis**: Quantified behavioral changes
- **Risk History**: Complete timeline of suspicious activities
- **Recommendation Engine**: Automated risk assessments

### Real-Time Scoring
- **Instant Risk Assessment**: Score new events in real-time
- **Feature Breakdown**: Detailed explanation of risk factors
- **Takeover Detection**: Specialized account compromise analysis
- **Interactive Interface**: User-friendly event input

## Performance Optimization

### Detection Quality
- **Precision-Focused**: Minimized false positives for security teams
- **Takeover Specialization**: Enhanced detection for account compromises
- **Scalable Processing**: Handles large-scale enterprise data
- **Real-Time Capability**: Sub-second scoring for new events

### Alert Management
- **Risk-Based Prioritization**: Focus on highest-impact threats
- **Explainable AI**: Clear reasoning for each alert
- **Actionable Intelligence**: Specific recommendations for investigation
- **Integration Ready**: API-compatible for SIEM integration

## Requirements

- Python 3.8+
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- streamlit >= 1.10.0
- plotly >= 5.0.0
- numpy >= 1.21.0

## Evolution Timeline

**Phase 1**: Basic Isolation Forest implementation
**Phase 2**: Enhanced behavioral features (22 features)
**Phase 3**: Ensemble models with weighted scoring
**Phase 4**: UEBA analytics with user baselines
**Phase 5**: Takeover-specific detection algorithms
**Phase 6**: Interactive dashboard with Plotly visualizations
**Current**: Optimized precision with ~1000 high-quality alerts

## Future Enhancements

- **Deep Learning Integration**: Neural network models for complex patterns
- **Real-Time Streaming**: Apache Kafka integration for live processing
- **Advanced Threat Intelligence**: External threat feed integration
- **Automated Response**: Integration with security orchestration platforms