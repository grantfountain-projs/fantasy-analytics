"""
Fantasy Football Outperformance Prediction Model

Machine learning model to predict which players will beat their projections by 20%+
Uses game context, player attributes, and historical patterns.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Dict, List, Tuple

class FantasyOutperformanceModel:
    def __init__(self, data_path="data/processed/fantasy_analysis_dataset.csv"):
        self.data_path = data_path
        self.model = None
        self.feature_columns = None
        self.label_encoders = {}
        self.model_metrics = {}
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        os.makedirs("analysis", exist_ok=True)
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare the dataset for modeling"""
        print("📊 Loading fantasy dataset...")
        df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Target distribution:")
        print(df['beat_projection'].value_counts(normalize=True))
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for the model"""
        print("🔧 Engineering features...")
        
        # Create copy to avoid modifying original
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
        
        print(f"Features created. New shape: {df.shape}")
        return df
    
    def prepare_model_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for modeling"""
        print("🎯 Preparing model data...")
        
        # Define feature columns
        feature_columns = [
            # Basic info
            'espn_projection', 'week', 'is_home',
            
            # Position features
            'is_qb', 'is_rb', 'is_wr', 'is_te',
            
            # Player tier
            'is_starter', 'is_backup',
            
            # Game context
            'spread', 'over_under', 'favorable_game_script', 'unfavorable_game_script',
            'large_spread', 'high_total', 'bad_weather', 'dome',
            
            # Interaction features
            'qb_high_total', 'rb_favorable_script', 'wr_high_total'
        ]
        
        # Handle categorical variables
        categorical_features = ['projection_tier']
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                feature_columns.append(f'{feature}_encoded')
                self.label_encoders[feature] = le
        
        # Prepare features and target
        X = df[feature_columns].fillna(0)  # Handle any missing values
        y = df['beat_projection'].astype(int)
        
        self.feature_columns = feature_columns
        
        print(f"Model features: {len(feature_columns)}")
        print(f"Feature columns: {feature_columns}")
        
        return X.values, y.values, feature_columns
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train multiple models and compare performance"""
        print("🤖 Training prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models to test
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=20,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        model_results = {}
        
        for name, model in models.items():
            print(f"\n🔍 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(X_test, y_test)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  AUC: {auc:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        return model_results
    
    def select_best_model(self, model_results: Dict) -> str:
        """Select the best performing model"""
        print("\n🏆 Model Comparison:")
        print("-" * 50)
        
        best_score = 0
        best_model = None
        
        for name, results in model_results.items():
            # Weighted score: AUC (70%) + CV accuracy (30%)
            score = 0.7 * results['auc'] + 0.3 * results['cv_mean']
            
            print(f"{name}:")
            print(f"  Accuracy: {results['accuracy']:.3f}")
            print(f"  AUC: {results['auc']:.3f}")
            print(f"  CV Score: {results['cv_mean']:.3f}")
            print(f"  Combined Score: {score:.3f}")
            print()
            
            if score > best_score:
                best_score = score
                best_model = name
        
        print(f"🥇 Best Model: {best_model} (Score: {best_score:.3f})")
        return best_model
    
    def analyze_feature_importance(self, model, feature_columns: List[str]):
        """Analyze and visualize feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🎯 Top 10 Most Important Features:")
            print("-" * 40)
            for idx, row in importance_df.head(10).iterrows():
                print(f"{row['feature']:25} {row['importance']:.3f}")
            
            # Create feature importance plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig('analysis/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return importance_df
        
        return None
    
    def evaluate_model_performance(self, model_results: Dict, best_model_name: str):
        """Detailed evaluation of the best model"""
        best_results = model_results[best_model_name]
        
        print(f"\n📊 Detailed Evaluation: {best_model_name}")
        print("=" * 50)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(best_results['y_test'], best_results['y_pred']))
        
        # Confusion matrix
        cm = confusion_matrix(best_results['y_test'], best_results['y_pred'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('analysis/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store metrics
        self.model_metrics = {
            'model_name': best_model_name,
            'accuracy': best_results['accuracy'],
            'auc': best_results['auc'],
            'cv_mean': best_results['cv_mean'],
            'cv_std': best_results['cv_std']
        }
    
    def save_model(self, model, model_name: str):
        """Save the trained model"""
        model_path = f"models/fantasy_outperformance_{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)
        
        # Save feature columns and encoders
        metadata = {
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'model_metrics': self.model_metrics
        }
        
        metadata_path = "models/model_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        
        print(f"💾 Model saved to: {model_path}")
        print(f"💾 Metadata saved to: {metadata_path}")
    
    def train_full_pipeline(self):
        """Run the complete training pipeline"""
        print("🚀 FANTASY OUTPERFORMANCE MODEL TRAINING")
        print("=" * 50)
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        
        # Create features
        df_features = self.create_features(df)
        
        # Prepare model data
        X, y, feature_columns = self.prepare_model_data(df_features)
        
        # Train models
        model_results = self.train_models(X, y)
        
        # Select best model
        best_model_name = self.select_best_model(model_results)
        best_model = model_results[best_model_name]['model']
        
        # Analyze feature importance
        importance_df = self.analyze_feature_importance(best_model, feature_columns)
        
        # Detailed evaluation
        self.evaluate_model_performance(model_results, best_model_name)
        
        # Save model
        self.save_model(best_model, best_model_name)
        self.model = best_model
        
        print(f"\n🎯 MODEL TRAINING COMPLETE!")
        print("=" * 50)
        print("✅ Model trained and saved")
        print("✅ Feature importance analyzed") 
        print("✅ Performance metrics calculated")
        print("✅ Ready for predictions!")
        
        return best_model, importance_df
    
    def predict_outperformance(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new player data"""
        if self.model is None:
            raise ValueError("Model not trained. Run train_full_pipeline() first.")
        
        # Apply same feature engineering
        player_features = self.create_features(player_data)
        
        # Select model features
        X = player_features[self.feature_columns].fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Add predictions to dataframe
        result = player_data.copy()
        result['predicted_outperform'] = predictions
        result['outperform_probability'] = probabilities
        result['confidence'] = np.where(probabilities > 0.5, probabilities, 1 - probabilities)
        
        return result


def main():
    """Main training execution"""
    # Initialize model trainer
    trainer = FantasyOutperformanceModel()
    
    # Train the complete pipeline
    model, feature_importance = trainer.train_full_pipeline()
    
    print(f"\n📈 NEXT STEPS:")
    print("1. Check analysis/ folder for visualizations")
    print("2. Use the model for weekly predictions")
    print("3. Build the web dashboard")
    print("4. Create prediction API endpoints")


if __name__ == "__main__":
    main()