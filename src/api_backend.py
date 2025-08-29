"""
FastAPI Backend for Fantasy Football Analytics Platform

Provides REST API endpoints for predictions, player data, and analytics.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Import our fixed predictor
import sys
sys.path.append('.')
from src.fixed_prediction_model import FantasyPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Fantasy Football Analytics API",
    description="ML-powered fantasy football predictions and analytics",
    version="1.0.0"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
predictor = None
players_df = None
analysis_df = None

@app.on_event("startup")
async def startup_event():
    """Load model and data on startup"""
    global predictor, players_df, analysis_df
    
    try:
        # Load predictor
        predictor = FantasyPredictor()
        print("✅ Model loaded successfully")
        
        # Load data
        analysis_df = pd.read_csv("data/processed/fantasy_analysis_dataset.csv")
        players_df = pd.read_csv("data/mock/mock_players.csv")
        print("✅ Data loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model/data: {e}")

# Pydantic models for request/response
class PlayerPredictionRequest(BaseModel):
    player_id: int
    week: int
    espn_projection: float
    position: str
    team_abbrev: str
    is_home: bool = True
    spread: float = 0.0
    over_under: float = 45.0
    weather_impact: bool = False
    dome: bool = False
    
class PredictionResponse(BaseModel):
    player_id: int
    name: str
    position: str
    team_abbrev: str
    espn_projection: float
    outperform_probability: float
    predicted_outperform: bool
    confidence: float

class WeeklyPredictionsResponse(BaseModel):
    week: int
    total_players: int
    predicted_outperformers: int
    predictions: List[PredictionResponse]

# API Endpoints

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Fantasy Football Analytics API",
        "status": "running",
        "version": "1.0.0",
        "model_loaded": predictor is not None,
        "data_loaded": analysis_df is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "api_status": "healthy",
        "model_loaded": predictor is not None,
        "data_rows": len(analysis_df) if analysis_df is not None else 0,
        "players_count": len(players_df) if players_df is not None else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/players")
async def get_players(
    position: Optional[str] = Query(None, description="Filter by position (QB, RB, WR, TE)"),
    team: Optional[str] = Query(None, description="Filter by team abbreviation"),
    limit: int = Query(50, description="Limit number of results")
):
    """Get player information with optional filtering"""
    if players_df is None:
        raise HTTPException(status_code=503, detail="Player data not loaded")
    
    df = players_df.copy()
    
    # Apply filters
    if position:
        df = df[df['position'] == position.upper()]
    
    if team:
        df = df[df['team_abbrev'] == team.upper()]
    
    # Limit results
    df = df.head(limit)
    
    return {
        "total_players": len(df),
        "filters": {"position": position, "team": team},
        "players": df.to_dict('records')
    }

@app.get("/weeks")
async def get_available_weeks():
    """Get list of available weeks for predictions"""
    if analysis_df is None:
        raise HTTPException(status_code=503, detail="Analysis data not loaded")
    
    weeks = sorted(analysis_df['week'].unique().tolist())
    
    return {
        "available_weeks": weeks,
        "total_weeks": len(weeks)
    }

@app.get("/predictions/week/{week}")
async def get_weekly_predictions(
    week: int,
    position: Optional[str] = Query(None, description="Filter by position"),
    team: Optional[str] = Query(None, description="Filter by team"),
    min_probability: float = Query(0.0, description="Minimum outperform probability"),
    limit: int = Query(100, description="Limit number of results")
) -> WeeklyPredictionsResponse:
    """Get predictions for a specific week"""
    if predictor is None or analysis_df is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    # Filter data for the week
    week_data = analysis_df[analysis_df['week'] == week].copy()
    
    if week_data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for week {week}")
    
    # Apply additional filters
    if position:
        week_data = week_data[week_data['position'] == position.upper()]
    
    if team:
        week_data = week_data[week_data['team_abbrev'] == team.upper()]
    
    # Make predictions
    try:
        predictions = predictor.predict_players(week_data)
        
        # Filter by minimum probability
        predictions = predictions[predictions['outperform_probability'] >= min_probability]
        
        # Sort by probability (highest first)
        predictions = predictions.sort_values('outperform_probability', ascending=False)
        
        # Limit results
        predictions = predictions.head(limit)
        
        # Convert to response format
        prediction_list = []
        for _, row in predictions.iterrows():
            prediction_list.append(PredictionResponse(
                player_id=int(row['player_id']),
                name=row['name'],
                position=row['position'],
                team_abbrev=row['team_abbrev'],
                espn_projection=float(row['espn_projection']),
                outperform_probability=float(row['outperform_probability']),
                predicted_outperform=bool(row['predicted_outperform']),
                confidence=float(row['confidence'])
            ))
        
        return WeeklyPredictionsResponse(
            week=week,
            total_players=len(week_data),
            predicted_outperformers=len([p for p in prediction_list if p.predicted_outperform]),
            predictions=prediction_list
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")

@app.get("/player/{player_id}/history")
async def get_player_history(player_id: int):
    """Get historical performance for a specific player"""
    if analysis_df is None:
        raise HTTPException(status_code=503, detail="Analysis data not loaded")
    
    player_data = analysis_df[analysis_df['player_id'] == player_id]
    
    if player_data.empty:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    
    # Calculate player statistics
    stats = {
        "player_id": player_id,
        "name": player_data['name'].iloc[0],
        "position": player_data['position'].iloc[0],
        "team": player_data['team_abbrev'].iloc[0],
        "games_played": len(player_data),
        "avg_projection": float(player_data['espn_projection'].mean()),
        "avg_actual": float(player_data['actual_points'].mean()),
        "beat_projection_rate": float(player_data['beat_projection'].mean()),
        "avg_outperformance_pct": float(player_data['outperformance_pct'].mean()),
        "weekly_performance": player_data[['week', 'espn_projection', 'actual_points', 'beat_projection', 'outperformance_pct']].to_dict('records')
    }
    
    return stats

@app.get("/analytics/position-performance")
async def get_position_performance():
    """Get performance analytics by position"""
    if analysis_df is None:
        raise HTTPException(status_code=503, detail="Analysis data not loaded")
    
    position_stats = analysis_df.groupby('position').agg({
        'beat_projection': ['count', 'sum', 'mean'],
        'outperformance_pct': 'mean',
        'espn_projection': 'mean',
        'actual_points': 'mean'
    }).round(3)
    
    # Flatten column names
    position_stats.columns = ['total_players', 'times_beat_projection', 'beat_rate', 'avg_outperformance_pct', 'avg_projection', 'avg_actual']
    
    # Convert to dict
    result = {}
    for position in position_stats.index:
        result[position] = {
            "total_players": int(position_stats.loc[position, 'total_players']),
            "times_beat_projection": int(position_stats.loc[position, 'times_beat_projection']),
            "beat_rate": float(position_stats.loc[position, 'beat_rate']),
            "avg_outperformance_pct": float(position_stats.loc[position, 'avg_outperformance_pct']),
            "avg_projection": float(position_stats.loc[position, 'avg_projection']),
            "avg_actual": float(position_stats.loc[position, 'avg_actual'])
        }
    
    return {"position_performance": result}

@app.get("/analytics/model-performance")
async def get_model_performance():
    """Get model performance metrics"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_metrics": predictor.metadata.get('model_metrics', {}),
        "feature_columns": predictor.feature_columns,
        "total_features": len(predictor.feature_columns)
    }

@app.post("/predict")
async def predict_single_player(request: PlayerPredictionRequest):
    """Make prediction for a single player with custom parameters"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Create DataFrame from request
    player_data = pd.DataFrame([{
        'player_id': request.player_id,
        'week': request.week,
        'espn_projection': request.espn_projection,
        'position': request.position,
        'team_abbrev': request.team_abbrev,
        'is_home': request.is_home,
        'spread': request.spread,
        'over_under': request.over_under,
        'weather_impact': request.weather_impact,
        'dome': request.dome,
        # Add required fields with defaults
        'tier': 'starter',
        'projected_game_script': 'neutral',
        'high_scoring_game': request.over_under > 48
    }])
    
    try:
        # Make prediction
        result = predictor.predict_players(player_data)
        
        return {
            "player_id": request.player_id,
            "outperform_probability": float(result['outperform_probability'].iloc[0]),
            "predicted_outperform": bool(result['predicted_outperform'].iloc[0]),
            "confidence": float(result['confidence'].iloc[0]),
            "input_parameters": request.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)