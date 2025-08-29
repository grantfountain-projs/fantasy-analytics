"""
Fixed Fantasy Prediction Model - handles feature engineering properly
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List
from sklearn.preprocessing import LabelEncoder

class FantasyPredictor:
    def __init__(self, model_path="models/fantasy_outperformance_random_forest.joblib", 
                 metadata_path="models/model_metadata.joblib"):
        """Initialize predictor with trained model"""
        self.model = joblib.load(model_path)
        self.metadata = joblib.load(metadata_path)
        self.feature_columns = self.metadata['feature_columns']
        self.label_encoders = self.metadata['label_encoders']
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the same features used during training"""
        df = df.copy()
        
        # Position-based features
        df['is_qb'] = (df['position'] == 'QB').astype(int)
        df['is_rb'] = (df['position'] == 'RB').astype(int) 
        df['is_wr'] = (df['position'] == 'WR').astype(int)
        df['is_te'] = (df['position'] == 'TE').astype(int)
        
        # Tier-based features
        df['is_starter'] = (df['tier'] == 'starter').astype(int)
        df['is_backup'] = (df['tier'] == 'backup').astype(int)
        
        # Game script features
        df['favorable_game_script'] = (df['projected_game_script'] == 'positive').astype(int)
        df['unfavorable_game_script'] = (df['projected_game_script'] == 'negative').astype(int)
        
        # Projection-based features
        df['projection_tier'] = pd.cut(df['espn_projection'], 
                                     bins=[0, 5, 10, 15, 20, 50], 
                                     labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Vegas line features
        df['large_spread'] = (abs(df['spread']) > 7).astype(int)
        df['high_total'] = (df['over_under'] > 48).astype(int)
        
        # Weather impact
        df['bad_weather'] = df['weather_impact'].astype(int)
        
        # Interaction features
        df['qb_high_total'] = df['is_qb'] * df['high_total']
        df['rb_favorable_script'] = df['is_rb'] * df['favorable_game_script']
        df['wr_high_total'] = df['is_wr'] * df['high_total']
        
        # Handle categorical encoding
        if 'projection_tier' in df.columns:
            # Use the saved label encoder
            if 'projection_tier' in self.label_encoders:
                le = self.label_encoders['projection_tier']
                df['projection_tier_encoded'] = le.transform(df['projection_tier'].astype(str))
            else:
                # Fallback encoding
                tier_mapping = {'very_low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
                df['projection_tier_encoded'] = df['projection_tier'].map(tier_mapping).fillna(2)
        
        return df
    
    def predict_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for a dataframe of players"""
        # Apply feature engineering
        df_with_features = self.create_features(df)
        
        # Select only the features the model was trained on
        try:
            X = df_with_features[self.feature_columns].fillna(0)
        except KeyError as e:
            print(f"Missing features: {e}")
            # Fill missing features with 0
            for col in self.feature_columns:
                if col not in df_with_features.columns:
                    df_with_features[col] = 0
            X = df_with_features[self.feature_columns].fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Add results to original dataframe
        result = df.copy()
        result['predicted_outperform'] = predictions
        result['outperform_probability'] = probabilities
        result['confidence'] = np.where(probabilities > 0.5, probabilities, 1 - probabilities)
        
        return result

def test_predictor():
    """Test the predictor with sample data"""
    try:
        predictor = FantasyPredictor()
        
        # Load test data
        df = pd.read_csv("data/processed/fantasy_analysis_dataset.csv")
        test_data = df.head(10)  # Test with first 10 players
        
        # Make predictions
        predictions = predictor.predict_players(test_data)
        
        print("✅ Predictor working!")
        print(f"Made predictions for {len(predictions)} players")
        print("\nSample predictions:")
        print(predictions[['name', 'position', 'espn_projection', 'outperform_probability']].head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_predictor()