"""
Fixed Fantasy Football Analytics Dashboard

Streamlit web application using the fixed predictor.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime
import sys

# Add src to path and import our fixed predictor
sys.path.append('src')
sys.path.append('.')
from src.fixed_prediction_model import FantasyPredictor

class FantasyDashboard:
    def __init__(self):
        self.predictor = None
        self.load_predictor()
        self.load_data()
    
    def load_predictor(self):
        """Load the fixed predictor"""
        try:
            self.predictor = FantasyPredictor()
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Model loading error: {e}")
            self.predictor = None
    
    def load_data(self):
        """Load the analysis dataset"""
        try:
            self.df = pd.read_csv("data/processed/fantasy_analysis_dataset.csv")
        except FileNotFoundError:
            st.error("❌ Data not found. Generate mock data first!")
            st.stop()
    
    def create_header(self):
        """Create dashboard header"""
        st.set_page_config(
            page_title="Fantasy Analytics Platform",
            page_icon="🏈",
            layout="wide"
        )
        
        st.title("🏈 Fantasy Football Analytics Platform")
        st.markdown("### Predict which players will outperform their projections by 20%+")
        
        # Model performance summary from the saved metadata
        try:
            metadata = joblib.load("models/model_metadata.joblib")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Accuracy", f"{metadata['model_metrics']['accuracy']:.1%}")
            with col2:
                st.metric("AUC Score", f"{metadata['model_metrics']['auc']:.3f}")
            with col3:
                st.metric("CV Score", f"{metadata['model_metrics']['cv_mean']:.3f}")
            with col4:
                st.metric("Total Players", f"{len(self.df):,}")
        except:
            st.info("Model metrics not available")
    
    def sidebar_controls(self):
        """Create sidebar controls"""
        st.sidebar.header("🎛️ Controls")
        
        # Week selector
        weeks = sorted(self.df['week'].unique())
        selected_week = st.sidebar.selectbox("Select Week", weeks, index=len(weeks)-1)
        
        # Position filter
        positions = ['All'] + sorted(self.df['position'].unique())
        selected_positions = st.sidebar.multiselect("Filter by Position", positions, default=['All'])
        
        if 'All' in selected_positions:
            selected_positions = self.df['position'].unique()
        
        # Team filter
        teams = ['All'] + sorted(self.df['team_abbrev'].unique())
        selected_teams = st.sidebar.multiselect("Filter by Team", teams, default=['All'])
        
        if 'All' in selected_teams:
            selected_teams = self.df['team_abbrev'].unique()
        
        return selected_week, selected_positions, selected_teams
    
    def filter_data(self, week, positions, teams):
        """Filter data based on selections"""
        filtered_df = self.df[
            (self.df['week'] == week) & 
            (self.df['position'].isin(positions)) & 
            (self.df['team_abbrev'].isin(teams))
        ].copy()
        
        return filtered_df
    
    def predictions_tab(self, df):
        """Create predictions tab"""
        st.header("🎯 Weekly Predictions")
        
        if self.predictor is None:
            st.error("❌ Predictor not loaded! Check the model files.")
            return
        
        try:
            # Make predictions using our fixed predictor
            predictions_df = self.predictor.predict_players(df)
            
            # Sort by outperformance probability
            predictions_df = predictions_df.sort_values('outperform_probability', ascending=False)
            
            # Display top predictions
            st.subheader("🌟 Top Outperformance Candidates")
            
            display_cols = [
                'name', 'position', 'team_abbrev', 'espn_projection', 
                'outperform_probability', 'predicted_outperform', 'actual_points', 'beat_projection'
            ]
            
            top_predictions = predictions_df[display_cols].head(20)
            
            # Format the dataframe for display
            top_predictions_display = top_predictions.copy()
            top_predictions_display['outperform_probability'] = top_predictions_display['outperform_probability'].apply(lambda x: f"{x:.1%}")
            top_predictions_display['predicted_outperform'] = top_predictions_display['predicted_outperform'].map({1: '✅ Yes', 0: '❌ No'})
            top_predictions_display['beat_projection'] = top_predictions_display['beat_projection'].map({True: '🎯 Yes', False: '❌ No'})
            
            top_predictions_display.columns = [
                'Player', 'Pos', 'Team', 'ESPN Proj', 'Outperform Prob', 'Predicted', 'Actual Pts', 'Actually Beat'
            ]
            
            st.dataframe(top_predictions_display, use_container_width=True)
            
            # Accuracy summary for this week
            correct_predictions = (predictions_df['predicted_outperform'] == predictions_df['beat_projection']).sum()
            total_predictions = len(predictions_df)
            accuracy = correct_predictions / total_predictions
            
            st.info(f"📊 Week Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions} correct)")
            
            # Show prediction distribution
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Players Predicted to Outperform", predictions_df['predicted_outperform'].sum())
            with col2:
                st.metric("Average Outperform Probability", f"{predictions_df['outperform_probability'].mean():.1%}")
            
        except Exception as e:
            st.error(f"❌ Error making predictions: {str(e)}")
            st.write("Debug info:")
            st.write(f"Data shape: {df.shape}")
            st.write(f"Data columns: {list(df.columns)}")
    
    def analysis_tab(self, df):
        """Create analysis tab"""
        st.header("📊 Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Outperformance by position
            position_stats = df.groupby('position').agg({
                'beat_projection': ['count', 'sum', 'mean'],
                'outperformance_pct': 'mean'
            }).round(3)
            
            position_stats.columns = ['Total Players', 'Beat Projection', 'Beat Rate', 'Avg Outperformance %']
            position_stats['Beat Rate'] = position_stats['Beat Rate'].apply(lambda x: f"{x:.1%}")
            position_stats['Avg Outperformance %'] = position_stats['Avg Outperformance %'].apply(lambda x: f"{x:+.1f}%")
            
            st.subheader("🏈 Performance by Position")
            st.dataframe(position_stats)
        
        with col2:
            # Projection accuracy scatter plot
            fig = px.scatter(
                df, 
                x='espn_projection', 
                y='actual_points',
                color='position',
                hover_data=['name', 'team_abbrev'],
                title="Projections vs Actual Performance"
            )
            
            # Add perfect prediction line
            max_val = max(df['espn_projection'].max(), df['actual_points'].max())
            fig.add_trace(
                go.Scatter(
                    x=[0, max_val], 
                    y=[0, max_val], 
                    mode='lines', 
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        st.subheader("🎯 What Drives Outperformance?")
        
        # Try to show feature importance from the model
        try:
            if hasattr(self.predictor.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': self.predictor.feature_columns,
                    'Importance': self.predictor.model.feature_importances_
                }).sort_values('Importance', ascending=True).tail(10)
                
                fig = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    title="Top 10 Most Important Prediction Features"
                )
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Feature importance chart not available")
        
        # Game context analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # High scoring games analysis
            if 'high_scoring_game' in df.columns:
                high_scoring = df.groupby('high_scoring_game')['beat_projection'].mean()
                
                fig = px.bar(
                    x=['Normal Games', 'High Scoring Games'],
                    y=[high_scoring[False], high_scoring[True]],
                    title="Outperformance Rate: High vs Normal Scoring Games"
                )
                fig.update_layout(yaxis_title="Beat Projection Rate")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Home vs away analysis
            if 'is_home' in df.columns:
                home_away = df.groupby('is_home')['beat_projection'].mean()
                
                fig = px.bar(
                    x=['Away Games', 'Home Games'],
                    y=[home_away[False], home_away[True]],
                    title="Outperformance Rate: Home vs Away Games"
                )
                fig.update_layout(yaxis_title="Beat Projection Rate")
                st.plotly_chart(fig, use_container_width=True)
    
    def insights_tab(self, df):
        """Create insights tab"""
        st.header("💡 Key Insights")
        
        # Calculate some insights
        insights = []
        
        # Best position for outperformance
        position_rates = df.groupby('position')['beat_projection'].mean()
        best_position = position_rates.idxmax()
        best_rate = position_rates.max()
        insights.append(f"**{best_position}s** have the highest outperformance rate at **{best_rate:.1%}**")
        
        # Game script insight
        if 'projected_game_script' in df.columns:
            game_script_rates = df.groupby('projected_game_script')['beat_projection'].mean()
            if len(game_script_rates) > 1:
                best_script = game_script_rates.idxmax()
                script_rate = game_script_rates.max()
                insights.append(f"Players in **{best_script}** game scripts outperform **{script_rate:.1%}** of the time")
        
        # Weather insight
        if 'weather_impact' in df.columns:
            weather_impact = df.groupby('weather_impact')['beat_projection'].mean()
            if len(weather_impact) > 1:
                weather_diff = weather_impact[True] - weather_impact[False]
                if abs(weather_diff) > 0.05:
                    direction = "more" if weather_diff > 0 else "less"
                    insights.append(f"Weather impact games see players outperform **{abs(weather_diff):.1%}** {direction} often")
        
        # High total games
        if 'high_scoring_game' in df.columns:
            total_impact = df.groupby('high_scoring_game')['beat_projection'].mean()
            if len(total_impact) > 1:
                total_diff = total_impact[True] - total_impact[False]
                if abs(total_diff) > 0.05:
                    direction = "more" if total_diff > 0 else "less"
                    insights.append(f"High-scoring games (O/U > 48) see **{abs(total_diff):.1%}** {direction} outperformance")
        
        # Display insights
        for i, insight in enumerate(insights, 1):
            st.write(f"{i}. {insight}")
        
        # Top consistent performers
        st.subheader("🌟 Most Consistent Outperformers")
        
        player_consistency = df.groupby(['name', 'position', 'team_abbrev']).agg({
            'beat_projection': ['count', 'sum', 'mean'],
            'outperformance_pct': 'mean'
        }).round(3)
        
        player_consistency.columns = ['Games', 'Times Beat', 'Beat Rate', 'Avg Outperformance %']
        
        # Filter players with multiple games
        consistent_players = player_consistency[player_consistency['Games'] >= 3].sort_values('Beat Rate', ascending=False).head(10)
        
        consistent_players['Beat Rate'] = consistent_players['Beat Rate'].apply(lambda x: f"{x:.1%}")
        consistent_players['Avg Outperformance %'] = consistent_players['Avg Outperformance %'].apply(lambda x: f"{x:+.1f}%")
        
        st.dataframe(consistent_players)
    
    def run_dashboard(self):
        """Run the complete dashboard"""
        self.create_header()
        
        # Sidebar controls
        selected_week, selected_positions, selected_teams = self.sidebar_controls()
        
        # Filter data
        filtered_df = self.filter_data(selected_week, selected_positions, selected_teams)
        
        st.markdown(f"**Showing {len(filtered_df)} players for Week {selected_week}**")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Predictions", "📊 Analysis", "💡 Insights"])
        
        with tab1:
            self.predictions_tab(filtered_df)
        
        with tab2:
            self.analysis_tab(filtered_df)
        
        with tab3:
            self.insights_tab(filtered_df)

def main():
    """Main dashboard execution"""
    dashboard = FantasyDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()