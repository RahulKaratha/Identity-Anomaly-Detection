import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

class UEBAVisualizations:
    """UEBA-style visualizations for SOC analysts"""
    
    def __init__(self):
        self.risk_colors = {
            'LOW': '#28a745',
            'MEDIUM': '#ffc107', 
            'HIGH': '#dc3545'
        }
    
    def show_user_risk_timeline(self, user_id, risk_progression):
        """Time-series visualization of user risk evolution with hover details"""
        if user_id not in risk_progression:
            st.warning(f"No data available for user {user_id}")
            return
        
        timeline = risk_progression[user_id]['timeline']
        df_timeline = pd.DataFrame(timeline)
        df_timeline['Login Timestamp'] = pd.to_datetime(df_timeline['Login Timestamp'])
        
        # Create the main risk score timeline with hover info
        fig = go.Figure()
        
        # Add risk score line with detailed hover information
        hover_text = []
        for _, row in df_timeline.iterrows():
            hover_info = f"""<b>Risk Score:</b> {row['risk_score']:.3f}<br>
<b>Risk Level:</b> {row['risk_level']}<br>
<b>Time:</b> {row['Login Timestamp'].strftime('%Y-%m-%d %H:%M')}<br>
<b>Flags:</b> {row.get('explanation', 'No specific flags')}<br>
<b>Trend:</b> {row.get('risk_trend', 0):.3f}"""
            hover_text.append(hover_info)
        
        fig.add_trace(go.Scatter(
            x=df_timeline['Login Timestamp'],
            y=df_timeline['risk_score'],
            mode='lines+markers',
            name='Risk Score',
            line=dict(color='blue', width=2),
            marker=dict(
                size=8,
                color=df_timeline['risk_score'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Risk Score")
            ),
            hovertemplate='%{text}<extra></extra>',
            text=hover_text
        ))
        
        # Add 7-day moving average
        fig.add_trace(go.Scatter(
            x=df_timeline['Login Timestamp'],
            y=df_timeline['risk_ma_7'],
            mode='lines',
            name='7-Day Average',
            line=dict(color='orange', width=2, dash='dash'),
            hovertemplate='7-Day Average: %{y:.3f}<extra></extra>'
        ))
        
        # Add risk level thresholds
        fig.add_hline(y=0.7, line_dash="dot", line_color="red", 
                     annotation_text="HIGH Risk Threshold")
        fig.add_hline(y=0.4, line_dash="dot", line_color="orange", 
                     annotation_text="MEDIUM Risk Threshold")
        
        # Highlight escalation points (simplified)
        escalations = risk_progression[user_id]['escalations']
        if escalations:
            st.write(f"**{len(escalations)} risk escalations detected**")
        
        # Color background by risk level
        for i, row in df_timeline.iterrows():
            color = self.risk_colors.get(row['risk_level'], '#f8f9fa')
            if i < len(df_timeline) - 1:
                fig.add_vrect(
                    x0=row['Login Timestamp'],
                    x1=df_timeline.iloc[i+1]['Login Timestamp'],
                    fillcolor=color,
                    opacity=0.1,
                    layer="below",
                    line_width=0
                )
        
        fig.update_layout(
            title=f"Risk Timeline for User {user_id} (Hover for Details)",
            xaxis_title="Time",
            yaxis_title="Risk Score",
            yaxis=dict(range=[0, 1]),
            height=500,
            showlegend=True,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show escalation summary with detailed flags
        if escalations:
            st.subheader("Risk Escalations with Flags")
            escalation_details = []
            for esc in escalations:
                escalation_details.append({
                    'Timestamp': pd.to_datetime(esc['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                    'Escalation': f"{esc['from_level']} → {esc['to_level']}",
                    'Risk Score': f"{esc['risk_score']:.3f}",
                    'Flags Raised': esc.get('explanation', 'No specific flags')
                })
            
            escalation_df = pd.DataFrame(escalation_details)
            st.dataframe(escalation_df, use_container_width=True)
    
    def show_behavioral_deviation_radar(self, user_id, deviations_df, df):
        """Radar chart showing behavioral deviations from baseline"""
        user_data = df[df['User ID'] == user_id]
        if len(user_data) == 0:
            return
        
        # Get recent deviations (last 10 events)
        recent_deviations = deviations_df.tail(10)
        avg_deviations = recent_deviations.mean()
        
        categories = ['Time Patterns', 'Geographic', 'Device Usage', 'Activity Level']
        values = [
            avg_deviations['time_deviation'],
            avg_deviations['geo_deviation'], 
            avg_deviations['device_deviation'],
            avg_deviations['activity_deviation']
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f'User {user_id}',
            line_color='red'
        ))
        
        # Add baseline (normal behavior)
        baseline_values = [0.2, 0.2, 0.2, 0.2]  # Normal deviation levels
        fig.add_trace(go.Scatterpolar(
            r=baseline_values,
            theta=categories,
            fill='toself',
            name='Normal Range',
            line_color='green',
            opacity=0.3
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=f"Behavioral Deviation Profile - User {user_id}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_user_cohort_analysis(self, df, user_baselines):
        """Show how user compares to similar user cohorts"""
        
        # Classify users into cohorts
        cohort_data = []
        for user_id, baseline in user_baselines.items():
            user_data = df[df['User ID'] == user_id]
            if len(user_data) == 0:
                continue
                
            cohort_data.append({
                'user_id': user_id,
                'profile_type': self._classify_user_profile(baseline),
                'avg_risk': user_data['risk_score'].mean(),
                'risk_volatility': user_data['risk_score'].std(),
                'total_events': len(user_data),
                'countries': len(baseline.get('countries', [])),
                'devices': len(baseline.get('devices', []))
            })
        
        cohort_df = pd.DataFrame(cohort_data)
        
        if len(cohort_df) == 0:
            return
        
        # Create cohort comparison visualization
        fig = px.scatter(
            cohort_df,
            x='avg_risk',
            y='risk_volatility',
            color='profile_type',
            size='total_events',
            hover_data=['user_id', 'countries', 'devices'],
            title="User Risk Profile Cohorts"
        )
        
        fig.update_layout(
            xaxis_title="Average Risk Score",
            yaxis_title="Risk Volatility",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show cohort statistics
        st.subheader("Cohort Statistics")
        cohort_stats = cohort_df.groupby('profile_type').agg({
            'avg_risk': ['mean', 'std'],
            'risk_volatility': 'mean',
            'user_id': 'count'
        }).round(3)
        
        st.dataframe(cohort_stats, use_container_width=True)
    
    def show_risk_escalation_heatmap(self, risk_progressions):
        """Heatmap showing risk escalation patterns across users"""
        
        escalation_data = []
        for user_id, progression in risk_progressions.items():
            escalations = progression.get('escalations', [])
            
            for escalation in escalations:
                # Ensure timestamp is datetime
                timestamp = pd.to_datetime(escalation['timestamp'])
                escalation_data.append({
                    'user_id': user_id,
                    'timestamp': timestamp,
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.strftime('%A'),
                    'escalation_type': f"{escalation['from_level']} → {escalation['to_level']}"
                })
        
        if not escalation_data:
            st.info("No risk escalations found in the data")
            return
        
        escalation_df = pd.DataFrame(escalation_data)
        
        # Create hour vs day of week heatmap
        heatmap_data = escalation_df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        fig = px.imshow(
            heatmap_data,
            title="Risk Escalation Patterns (Hour vs Day of Week)",
            labels=dict(x="Hour of Day", y="Day of Week", color="Escalations"),
            aspect="auto"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_user_journey_flow(self, user_id, df):
        """Sankey diagram showing user's risk level transitions"""
        user_data = df[df['User ID'] == user_id].sort_values('Login Timestamp')
        
        if len(user_data) < 2:
            st.warning("Insufficient data for journey flow")
            return
        
        # Create transitions
        transitions = []
        for i in range(len(user_data) - 1):
            current_level = user_data.iloc[i]['risk_level']
            next_level = user_data.iloc[i + 1]['risk_level']
            transitions.append(f"{current_level} → {next_level}")
        
        transition_counts = pd.Series(transitions).value_counts()
        
        # Create Sankey diagram data
        levels = ['LOW', 'MEDIUM', 'HIGH']
        source = []
        target = []
        value = []
        
        for transition, count in transition_counts.items():
            from_level, to_level = transition.split(' → ')
            source.append(levels.index(from_level))
            target.append(levels.index(to_level) + 3)  # Offset target nodes
            value.append(count)
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=levels + [f"{level} (Next)" for level in levels],
                color=["green", "orange", "red", "lightgreen", "lightyellow", "lightcoral"]
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])
        
        fig.update_layout(
            title_text=f"Risk Level Transitions - User {user_id}",
            font_size=10
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
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