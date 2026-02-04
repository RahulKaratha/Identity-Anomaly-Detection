import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class UserBehavioralAnalytics:
    """UEBA-style user behavioral analytics and profiling"""
    
    def __init__(self, lookback_days=30, min_events=10):
        self.lookback_days = lookback_days
        self.min_events = min_events
        self.user_profiles = {}
        
    def build_user_baselines(self, df):
        """Build personalized behavioral baselines for each user"""
        user_baselines = {}
        
        # Ensure timestamp is datetime
        df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
        
        for user_id in df['User ID'].unique():
            user_data = df[df['User ID'] == user_id].copy()
            
            if len(user_data) < self.min_events:
                continue
                
            # Time-based patterns
            login_hours = user_data['login_hour'].values
            login_days = user_data['login_day'].values if 'login_day' in user_data.columns else user_data['Login Timestamp'].dt.dayofweek.values
            
            # Calculate date range safely
            date_range = max(1, (user_data['Login Timestamp'].max() - user_data['Login Timestamp'].min()).days)
            
            baseline = {
                # Time patterns
                'typical_hours': self._get_typical_range(login_hours),
                'typical_days': self._get_typical_range(login_days),
                'weekend_user': (user_data.get('is_weekend', 0).mean() > 0.3),
                
                # Geographic patterns
                'countries': set(user_data['Country'].unique()) if 'Country' in user_data.columns else {'Unknown'},
                'primary_country': user_data['Country'].mode().iloc[0] if 'Country' in user_data.columns and len(user_data) > 0 else 'Unknown',
                'is_traveler': len(user_data['Country'].unique()) > 2 if 'Country' in user_data.columns else False,
                
                # Device patterns
                'devices': set(user_data['Device Type'].unique()) if 'Device Type' in user_data.columns else {'Unknown'},
                'primary_device': user_data['Device Type'].mode().iloc[0] if 'Device Type' in user_data.columns and len(user_data) > 0 else 'Unknown',
                'multi_device_user': len(user_data['Device Type'].unique()) > 1 if 'Device Type' in user_data.columns else False,
                
                # Activity patterns
                'avg_daily_logins': len(user_data) / date_range,
                'failure_rate': user_data.get('login_failed', 0).mean(),
                'avg_rtt': user_data['Round-Trip Time [ms]'].mean() if 'Round-Trip Time [ms]' in user_data.columns else 200,
                
                # Risk history
                'baseline_risk': user_data['risk_score'].quantile(0.75) if 'risk_score' in user_data.columns else 0.3,
                'risk_volatility': user_data['risk_score'].std() if 'risk_score' in user_data.columns else 0.1,
                
                # Metadata
                'profile_created': datetime.now(),
                'total_events': len(user_data),
                'first_seen': user_data['Login Timestamp'].min(),
                'last_seen': user_data['Login Timestamp'].max()
            }
            
            user_baselines[user_id] = baseline
            
        return user_baselines
    
    def _get_typical_range(self, values, percentile_range=(25, 75)):
        """Get typical range for a behavioral metric"""
        if len(values) == 0:
            return (0, 23)
        return (np.percentile(values, percentile_range[0]), np.percentile(values, percentile_range[1]))
    
    def compute_behavioral_deviations(self, df, user_baselines):
        """Compute how much each event deviates from user's baseline"""
        deviations = []
        
        for _, row in df.iterrows():
            user_id = row['User ID']
            baseline = user_baselines.get(user_id, {})
            
            if not baseline:
                # New user - use global defaults
                deviation = {
                    'time_deviation': 0.5,
                    'geo_deviation': 0.3,
                    'device_deviation': 0.2,
                    'activity_deviation': 0.3,
                    'overall_deviation': 0.4
                }
            else:
                # Time deviation
                hour_in_range = baseline['typical_hours'][0] <= row.get('login_hour', 12) <= baseline['typical_hours'][1]
                day_in_range = baseline['typical_days'][0] <= row.get('login_day', 2) <= baseline['typical_days'][1]
                weekend_expected = baseline['weekend_user'] and row.get('is_weekend', 0)
                
                time_dev = 0.0
                if not hour_in_range:
                    time_dev += 0.6
                if not day_in_range and not weekend_expected:
                    time_dev += 0.4
                
                # Geographic deviation
                geo_dev = 0.0
                user_country = row.get('Country', 'Unknown')
                if user_country not in baseline['countries']:
                    geo_dev = 0.3 if baseline['is_traveler'] else 0.8
                elif user_country != baseline['primary_country']:
                    geo_dev = 0.1 if baseline['is_traveler'] else 0.4
                
                # Device deviation
                device_dev = 0.0
                user_device = row.get('Device Type', 'Unknown')
                if user_device not in baseline['devices']:
                    device_dev = 0.2 if baseline['multi_device_user'] else 0.7
                elif user_device != baseline['primary_device']:
                    device_dev = 0.1 if baseline['multi_device_user'] else 0.3
                
                # Activity deviation (failure rate, timing)
                activity_dev = 0.0
                if row.get('login_failed', 0) and baseline['failure_rate'] < 0.1:
                    activity_dev += 0.5
                
                # Overall deviation score
                overall_dev = np.mean([time_dev, geo_dev, device_dev, activity_dev])
                
                deviation = {
                    'time_deviation': time_dev,
                    'geo_deviation': geo_dev,
                    'device_deviation': device_dev,
                    'activity_deviation': activity_dev,
                    'overall_deviation': overall_dev
                }
            
            deviations.append(deviation)
        
        return pd.DataFrame(deviations)
    
    def track_risk_progression(self, df):
        """Track how user risk scores evolve over time"""
        risk_progressions = {}
        
        # Ensure timestamp is datetime
        df = df.copy()
        df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
        
        for user_id in df['User ID'].unique():
            user_data = df[df['User ID'] == user_id].copy()
            user_data = user_data.sort_values('Login Timestamp')
            
            if len(user_data) < 2:
                continue
            
            # Calculate rolling statistics
            user_data['risk_ma_7'] = user_data['risk_score'].rolling(window=min(7, len(user_data)), min_periods=1).mean()
            user_data['risk_trend'] = user_data['risk_score'].rolling(window=min(5, len(user_data)), min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            
            # Identify escalation events with explanations
            escalations = []
            prev_level = 'LOW'
            
            for _, row in user_data.iterrows():
                current_level = row['risk_level']
                if self._is_escalation(prev_level, current_level):
                    escalations.append({
                        'timestamp': row['Login Timestamp'],
                        'from_level': prev_level,
                        'to_level': current_level,
                        'risk_score': row['risk_score'],
                        'explanation': row.get('explanation', 'Behavioral deviation detected')
                    })
                prev_level = current_level
            
            # Risk stability metrics
            recent_data = user_data.tail(min(20, len(user_data)))
            stability_score = 1 / (1 + recent_data['risk_score'].std())
            
            risk_progressions[user_id] = {
                'timeline': user_data[['Login Timestamp', 'risk_score', 'risk_level', 'risk_ma_7', 'risk_trend', 'explanation']].to_dict('records'),
                'escalations': escalations,
                'current_trend': user_data['risk_trend'].iloc[-1] if len(user_data) > 0 else 0,
                'stability_score': stability_score,
                'max_risk_score': user_data['risk_score'].max(),
                'avg_risk_score': user_data['risk_score'].mean(),
                'days_since_high_risk': self._days_since_high_risk(user_data),
                'escalation_frequency': len(escalations) / max(1, len(user_data) / 30)  # per month
            }
        
        return risk_progressions
    
    def _is_escalation(self, prev_level, current_level):
        """Check if there's a risk level escalation"""
        levels = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        return levels.get(current_level, 0) > levels.get(prev_level, 0)
    
    def _days_since_high_risk(self, user_data):
        """Calculate days since last HIGH risk event"""
        high_risk_events = user_data[user_data['risk_level'] == 'HIGH']
        if len(high_risk_events) == 0:
            return float('inf')
        
        last_high_risk = high_risk_events['Login Timestamp'].max()
        return (datetime.now() - last_high_risk).days
    
    def generate_user_risk_summary(self, user_id, df, user_baselines, risk_progressions):
        """Generate comprehensive risk summary for a user"""
        user_data = df[df['User ID'] == user_id].copy()
        baseline = user_baselines.get(user_id, {})
        progression = risk_progressions.get(user_id, {})
        
        if len(user_data) == 0:
            return None
        
        # Ensure timestamp is datetime
        user_data['Login Timestamp'] = pd.to_datetime(user_data['Login Timestamp'])
        
        recent_data = user_data.tail(10)
        
        # Calculate days active safely
        date_range = (user_data['Login Timestamp'].max() - user_data['Login Timestamp'].min()).days
        
        summary = {
            'user_id': user_id,
            'profile_type': self._classify_user_profile(baseline),
            'current_risk_level': recent_data['risk_level'].iloc[-1],
            'current_risk_score': recent_data['risk_score'].iloc[-1],
            'risk_trend': progression.get('current_trend', 0),
            'stability_score': progression.get('stability_score', 0.5),
            'days_active': max(1, date_range),
            'total_events': len(user_data),
            'recent_escalations': len([e for e in progression.get('escalations', []) 
                                     if (datetime.now() - e['timestamp']).days <= 7]),
            'behavioral_flags': self._get_behavioral_flags(user_data, baseline),
            'recommendation': self._get_security_recommendation(user_data, baseline, progression)
        }
        
        return summary
    
    def _classify_user_profile(self, baseline):
        """Classify user into behavioral profile types"""
        if not baseline:
            return 'New User'
        
        if baseline.get('is_traveler', False) and baseline.get('multi_device_user', False):
            return 'Mobile Executive'
        elif baseline.get('weekend_user', False):
            return 'Shift Worker'
        elif baseline.get('avg_daily_logins', 0) > 10:
            return 'Power User'
        elif baseline.get('failure_rate', 0) > 0.2:
            return 'Struggling User'
        else:
            return 'Regular User'
    
    def _get_behavioral_flags(self, user_data, baseline):
        """Get current behavioral flags for the user"""
        flags = []
        recent_data = user_data.tail(5)
        
        if recent_data['country_change'].sum() > 2:
            flags.append('Frequent Location Changes')
        
        if recent_data['device_change'].sum() > 1:
            flags.append('Multiple Device Changes')
        
        if recent_data['login_failed'].mean() > baseline.get('failure_rate', 0.1) * 2:
            flags.append('Increased Login Failures')
        
        if recent_data['is_night'].mean() > 0.5 and not baseline.get('weekend_user', False):
            flags.append('Unusual Time Activity')
        
        return flags
    
    def _get_security_recommendation(self, user_data, baseline, progression):
        """Generate security recommendation based on user behavior"""
        recent_risk = user_data.tail(5)['risk_score'].mean()
        trend = progression.get('current_trend', 0)
        escalations = len(progression.get('escalations', []))
        
        if recent_risk > 0.7 and trend > 0.1:
            return 'IMMEDIATE REVIEW - High risk with increasing trend'
        elif escalations > 3:
            return 'MONITOR CLOSELY - Frequent risk escalations'
        elif recent_risk > 0.5 and trend > 0.05:
            return 'INVESTIGATE - Moderate risk with upward trend'
        elif recent_risk < 0.3 and trend < 0.01:
            return 'NORMAL - Stable low-risk behavior'
        else:
            return 'ROUTINE MONITORING - Standard risk profile'