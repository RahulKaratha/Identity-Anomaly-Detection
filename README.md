# Identity Anomaly Detection

A machine learning system that detects suspicious login behavior using the Isolation Forest algorithm. The system analyzes user login patterns and flags potentially compromised accounts based on behavioral anomalies.

## What it does

This tool monitors user login activities and identifies unusual patterns that might indicate account takeovers or security threats. It looks at factors like:

- Login times and days
- Device changes
- Geographic location changes
- Failed login attempts
- Network latency patterns

## How to use

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data**
   - Place your login data in `data/login_data.csv`
   - The data should include columns: User ID, Login Timestamp, Device Type, Country, Login Successful, Round-Trip Time [ms]

3. **Run the analysis**
   ```bash
   # Process the raw data
   python src/preprocess.py
   
   # Train the model and generate alerts
   python src/model.py
   ```

4. **View results**
   ```bash
   # Launch the dashboard
   streamlit run dashboard/dashboard.py
   ```

## Project structure

```
├── src/
│   ├── preprocess.py    # Cleans and prepares login data
│   └── model.py         # Trains Isolation Forest and scores events
├── dashboard/
│   └── dashboard.py     # Interactive web dashboard
├── data/
│   ├── login_data.csv          # Raw login events
│   ├── preprocessed_logs.csv   # Cleaned data with features
│   └── alerts_ready.csv        # Final results with risk scores
└── requirements.txt
```

## How it works

The system uses an **Isolation Forest** algorithm, which is particularly good at finding outliers in data. Here's the process:

1. **Feature extraction**: Creates behavioral features from raw login data
2. **Model training**: Learns normal login patterns using unsupervised learning
3. **Anomaly scoring**: Assigns risk scores to each login event
4. **Alert generation**: Categorizes events as LOW, MEDIUM, or HIGH risk

## Key features

- **Unsupervised learning**: No need for labeled training data
- **Real-time scoring**: Can process new login events as they happen
- **Explainable alerts**: Each alert comes with reasons why it was flagged
- **Interactive dashboard**: Easy-to-use web interface for security teams
- **Configurable thresholds**: Adjust sensitivity based on your security needs

## Future improvements

- **Enhanced features**: Add more behavioral indicators like typing patterns, session duration
- **Real-time processing**: Stream processing for immediate threat detection
- **Machine learning improvements**: Experiment with other anomaly detection algorithms
- **Integration capabilities**: APIs for SIEM systems and security tools
- **User feedback loop**: Allow security analysts to mark false positives to improve accuracy
- **Geographic intelligence**: Better location-based risk assessment
- **Time series analysis**: Detect gradual changes in user behavior over time
- **Multi-factor correlation**: Combine multiple risk signals for better accuracy

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- streamlit
- matplotlib
- numpy