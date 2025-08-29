"""
Fantasy Football Mock Data Generator

Creates realistic fantasy football data for platform development and demo.
Generates player projections, actual performance, and game context data.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
import json

class FantasyMockDataGenerator:
    def __init__(self):
        self.data_dir = "data/mock"
        self.processed_dir = "data/processed"
        self.ensure_directories()
        
        # NFL teams for 2024 season
        self.nfl_teams = [
            {'name': 'Kansas City Chiefs', 'abbrev': 'KC', 'conference': 'AFC'},
            {'name': 'Philadelphia Eagles', 'abbrev': 'PHI', 'conference': 'NFC'},
            {'name': 'Buffalo Bills', 'abbrev': 'BUF', 'conference': 'AFC'},
            {'name': 'Dallas Cowboys', 'abbrev': 'DAL', 'conference': 'NFC'},
            {'name': 'San Francisco 49ers', 'abbrev': 'SF', 'conference': 'NFC'},
            {'name': 'Miami Dolphins', 'abbrev': 'MIA', 'conference': 'AFC'},
            {'name': 'Cincinnati Bengals', 'abbrev': 'CIN', 'conference': 'AFC'},
            {'name': 'New York Jets', 'abbrev': 'NYJ', 'conference': 'AFC'},
            {'name': 'Minnesota Vikings', 'abbrev': 'MIN', 'conference': 'NFC'},
            {'name': 'Jacksonville Jaguars', 'abbrev': 'JAX', 'conference': 'AFC'},
            {'name': 'Green Bay Packers', 'abbrev': 'GB', 'conference': 'NFC'},
            {'name': 'Detroit Lions', 'abbrev': 'DET', 'conference': 'NFC'},
            {'name': 'Seattle Seahawks', 'abbrev': 'SEA', 'conference': 'NFC'},
            {'name': 'Los Angeles Chargers', 'abbrev': 'LAC', 'conference': 'AFC'},
            {'name': 'Baltimore Ravens', 'abbrev': 'BAL', 'conference': 'AFC'},
            {'name': 'Cleveland Browns', 'abbrev': 'CLE', 'conference': 'AFC'},
            {'name': 'Tampa Bay Buccaneers', 'abbrev': 'TB', 'conference': 'NFC'},
            {'name': 'Las Vegas Raiders', 'abbrev': 'LV', 'conference': 'AFC'},
            {'name': 'Atlanta Falcons', 'abbrev': 'ATL', 'conference': 'NFC'},
            {'name': 'Pittsburgh Steelers', 'abbrev': 'PIT', 'conference': 'AFC'},
            {'name': 'Los Angeles Rams', 'abbrev': 'LAR', 'conference': 'NFC'},
            {'name': 'Houston Texans', 'abbrev': 'HOU', 'conference': 'AFC'},
            {'name': 'New York Giants', 'abbrev': 'NYG', 'conference': 'NFC'},
            {'name': 'Indianapolis Colts', 'abbrev': 'IND', 'conference': 'AFC'},
            {'name': 'Tennessee Titans', 'abbrev': 'TEN', 'conference': 'AFC'},
            {'name': 'Washington Commanders', 'abbrev': 'WSH', 'conference': 'NFC'},
            {'name': 'New Orleans Saints', 'abbrev': 'NO', 'conference': 'NFC'},
            {'name': 'Chicago Bears', 'abbrev': 'CHI', 'conference': 'NFC'},
            {'name': 'Denver Broncos', 'abbrev': 'DEN', 'conference': 'AFC'},
            {'name': 'Carolina Panthers', 'abbrev': 'CAR', 'conference': 'NFC'},
            {'name': 'Arizona Cardinals', 'abbrev': 'ARI', 'conference': 'NFC'},
            {'name': 'New England Patriots', 'abbrev': 'NE', 'conference': 'AFC'}
        ]
        
        # Realistic player names by position
        self.player_names = {
            'QB': ['Josh Allen', 'Lamar Jackson', 'Dak Prescott', 'Tua Tagovailoa', 'Russell Wilson',
                  'Aaron Rodgers', 'Kirk Cousins', 'Trevor Lawrence', 'Justin Herbert', 'Jalen Hurts'],
            'RB': ['Christian McCaffrey', 'Austin Ekeler', 'Derrick Henry', 'Nick Chubb', 'Jonathan Taylor',
                  'Saquon Barkley', 'Alvin Kamara', 'Dalvin Cook', 'Joe Mixon', 'Aaron Jones'],
            'WR': ['Cooper Kupp', 'Davante Adams', 'Stefon Diggs', 'Tyreek Hill', 'DeAndre Hopkins',
                  'Mike Evans', 'Keenan Allen', 'A.J. Brown', 'DK Metcalf', 'CeeDee Lamb'],
            'TE': ['Travis Kelce', 'Mark Andrews', 'George Kittle', 'Darren Waller', 'T.J. Hockenson',
                  'Kyle Pitts', 'Dallas Goedert', 'Pat Freiermuth', 'David Njoku', 'Evan Engram']
        }
    
    def ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def generate_players(self) -> List[Dict]:
        """Generate realistic NFL player roster"""
        players = []
        player_id = 1000
        
        for team in self.nfl_teams:
            # Generate players for each position per team
            positions = {
                'QB': (2, 3),    # 2-3 QBs per team
                'RB': (3, 5),    # 3-5 RBs per team  
                'WR': (5, 7),    # 5-7 WRs per team
                'TE': (2, 3)     # 2-3 TEs per team
            }
            
            for position, (min_count, max_count) in positions.items():
                player_count = random.randint(min_count, max_count)
                
                for i in range(player_count):
                    # Create realistic player
                    base_name = random.choice(self.player_names[position])
                    # Modify name slightly to create unique players
                    if i > 0:
                        suffixes = ['Jr.', 'II', 'III', 'Sr.']
                        first_names = ['Marcus', 'DeShawn', 'Tyler', 'Brandon', 'Jordan', 'Michael']
                        if random.random() < 0.3:
                            name_parts = base_name.split()
                            name_parts[0] = random.choice(first_names)
                            player_name = ' '.join(name_parts)
                        else:
                            player_name = f"{base_name} {random.choice(suffixes)}" if random.random() < 0.2 else base_name
                    else:
                        player_name = base_name
                    
                    # Generate player attributes
                    experience = random.randint(1, 12)  # Years in NFL
                    age = 22 + experience + random.randint(-2, 2)
                    
                    player = {
                        'player_id': player_id,
                        'name': player_name,
                        'position': position,
                        'team_name': team['name'],
                        'team_abbrev': team['abbrev'],
                        'conference': team['conference'],
                        'age': age,
                        'experience': experience,
                        'jersey_number': random.randint(1, 99),
                        'tier': self.assign_player_tier(position, i)  # Starter, backup, etc.
                    }
                    
                    players.append(player)
                    player_id += 1
        
        return players
    
    def assign_player_tier(self, position: str, index: int) -> str:
        """Assign player tier based on position and depth chart"""
        if position == 'QB':
            return 'starter' if index == 0 else 'backup'
        elif position in ['RB', 'WR']:
            if index == 0:
                return 'starter'
            elif index == 1:
                return 'rb2' if position == 'RB' else 'wr2'
            else:
                return 'depth'
        else:  # TE
            return 'starter' if index == 0 else 'backup'
    
    def generate_week_projections(self, players: List[Dict], week: int) -> List[Dict]:
        """Generate fantasy projections for a specific week"""
        projections = []
        
        # Projection ranges by position and tier
        projection_ranges = {
            'QB': {'starter': (18, 28), 'backup': (8, 18)},
            'RB': {'starter': (12, 22), 'rb2': (8, 15), 'depth': (2, 8)},
            'WR': {'starter': (10, 20), 'wr2': (6, 14), 'depth': (2, 8)},
            'TE': {'starter': (6, 14), 'backup': (2, 6)}
        }
        
        for player in players:
            position = player['position']
            tier = player['tier']
            
            # Get projection range
            min_proj, max_proj = projection_ranges[position][tier]
            
            # Add some randomness and weekly variance
            base_projection = random.uniform(min_proj, max_proj)
            weekly_variance = random.uniform(-0.2, 0.2)  # ±20% variance
            projection = base_projection * (1 + weekly_variance)
            
            # Add matchup factors (some players have better/worse matchups)
            matchup_factor = random.uniform(0.8, 1.2)  # 80% to 120% of base
            final_projection = projection * matchup_factor
            
            projections.append({
                'week': week,
                'player_id': player['player_id'],
                'name': player['name'],
                'position': player['position'],
                'team_abbrev': player['team_abbrev'],
                'tier': player['tier'],
                'espn_projection': round(final_projection, 1),
                'yahoo_projection': round(final_projection * random.uniform(0.9, 1.1), 1),  # Slight difference
                'matchup_factor': round(matchup_factor, 2),
                'projection_confidence': random.uniform(0.6, 0.95)
            })
        
        return projections
    
    def generate_actual_performance(self, projections: List[Dict]) -> List[Dict]:
        """Generate actual fantasy points (what really happened)"""
        actuals = []
        
        for proj in projections:
            espn_proj = proj['espn_projection']
            
            # Create realistic variance from projections
            # Some players consistently outperform, others underperform
            player_tendency = random.uniform(-0.3, 0.3)  # Player's tendency to over/under perform
            
            # Weekly randomness (injuries, game script, etc.)
            weekly_randomness = np.random.normal(0, 0.4)  # Normal distribution
            
            # Game script factors (blowouts, close games, weather, etc.)
            game_script_factor = random.choice([
                0.6,   # Blowout (limited touches)
                0.8,   # Negative game script
                1.0,   # Normal game
                1.2,   # Positive game script
                1.4    # Shootout/high-scoring
            ])
            
            # Calculate actual points
            performance_multiplier = (1 + player_tendency + weekly_randomness) * game_script_factor
            actual_points = max(0, espn_proj * performance_multiplier)  # Can't be negative
            
            # Determine if they beat projection significantly (our target variable)
            beat_projection = actual_points > (espn_proj * 1.2)  # 20% better than expected
            outperformance_pct = ((actual_points - espn_proj) / espn_proj) * 100
            
            actuals.append({
                'week': proj['week'],
                'player_id': proj['player_id'],
                'name': proj['name'],
                'position': proj['position'],
                'team_abbrev': proj['team_abbrev'],
                'tier': proj['tier'],
                'espn_projection': espn_proj,
                'actual_points': round(actual_points, 1),
                'beat_projection': beat_projection,
                'outperformance_pct': round(outperformance_pct, 1),
                'game_script_factor': game_script_factor,
                'player_tendency': round(player_tendency, 3)
            })
        
        return actuals
    
    def generate_game_context(self, week: int) -> List[Dict]:
        """Generate game context data (spreads, over/unders, weather)"""
        games = []
        game_id = 1
        
        # Create matchups (16 games per week)
        teams = self.nfl_teams.copy()
        random.shuffle(teams)
        
        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                home_team = teams[i]
                away_team = teams[i + 1]
                
                # Generate realistic betting data
                spread = random.uniform(-14, 14)  # Point spread
                over_under = random.uniform(38, 55)  # Total points
                
                # Weather factors
                weather_impact = random.choice(['None', 'Wind', 'Rain', 'Snow', 'Heat', 'Cold'])
                
                games.append({
                    'week': week,
                    'game_id': f'game_{week}_{game_id}',
                    'home_team': home_team['abbrev'],
                    'away_team': away_team['abbrev'],
                    'spread': round(spread, 1),
                    'over_under': round(over_under, 1),
                    'weather': weather_impact,
                    'game_type': random.choice(['divisional', 'conference', 'interconference']),
                    'dome': random.choice([True, False])
                })
                
                game_id += 1
        
        return games
    
    def generate_full_dataset(self, weeks: int = 8) -> Dict:
        """Generate complete mock dataset for multiple weeks"""
        print(f"🎯 GENERATING MOCK FANTASY DATASET")
        print(f"Weeks: {weeks}, Players: ~{len(self.nfl_teams) * 15}")
        print("=" * 50)
        
        # Generate players
        print("👥 Generating players...")
        players = self.generate_players()
        
        # Generate data for each week
        all_projections = []
        all_actuals = []
        all_games = []
        
        for week in range(1, weeks + 1):
            print(f"📊 Generating Week {week} data...")
            
            # Weekly projections
            week_projections = self.generate_week_projections(players, week)
            all_projections.extend(week_projections)
            
            # Actual performance
            week_actuals = self.generate_actual_performance(week_projections)
            all_actuals.extend(week_actuals)
            
            # Game context
            week_games = self.generate_game_context(week)
            all_games.extend(week_games)
        
        return {
            'players': players,
            'projections': all_projections,
            'actuals': all_actuals,
            'games': all_games
        }
    
    def save_datasets(self, data: Dict):
        """Save all generated datasets"""
        print("\n💾 Saving datasets...")
        
        # Save individual files
        datasets = {
            'players': 'mock_players.csv',
            'projections': 'mock_projections.csv', 
            'actuals': 'mock_actual_performance.csv',
            'games': 'mock_game_context.csv'
        }
        
        for data_type, filename in datasets.items():
            df = pd.DataFrame(data[data_type])
            filepath = f"{self.data_dir}/{filename}"
            df.to_csv(filepath, index=False)
            print(f"  ✅ {filepath} ({len(df)} records)")
        
        # Create master analytical dataset
        self.create_analysis_dataset(data)
    
    def create_analysis_dataset(self, data: Dict):
        """Create master dataset for analysis and modeling"""
        print("\n🔬 Creating analysis dataset...")
        
        # Merge all data together
        actuals_df = pd.DataFrame(data['actuals'])
        games_df = pd.DataFrame(data['games'])
        
        # Create analysis dataset with features for modeling
        analysis_data = []
        
        for _, actual in actuals_df.iterrows():
            week = actual['week']
            team = actual['team_abbrev']
            
            # Find matching game context
            game_context = games_df[
                (games_df['week'] == week) & 
                ((games_df['home_team'] == team) | (games_df['away_team'] == team))
            ]
            
            if not game_context.empty:
                game = game_context.iloc[0]
                is_home = game['home_team'] == team
                
                analysis_record = {
                    # Basic info
                    'week': week,
                    'player_id': actual['player_id'],
                    'name': actual['name'],
                    'position': actual['position'],
                    'team_abbrev': team,
                    'tier': actual['tier'],
                    
                    # Performance data
                    'espn_projection': actual['espn_projection'],
                    'actual_points': actual['actual_points'],
                    'beat_projection': actual['beat_projection'],
                    'outperformance_pct': actual['outperformance_pct'],
                    
                    # Game context features
                    'is_home': is_home,
                    'spread': game['spread'] if is_home else -game['spread'],
                    'over_under': game['over_under'],
                    'weather': game['weather'],
                    'dome': game['dome'],
                    'game_type': game['game_type'],
                    
                    # Derived features
                    'projected_game_script': 'positive' if (game['spread'] > 3 and is_home) or (game['spread'] < -3 and not is_home) else 'negative' if (game['spread'] < -3 and is_home) or (game['spread'] > 3 and not is_home) else 'neutral',
                    'high_scoring_game': game['over_under'] > 48,
                    'weather_impact': game['weather'] not in ['None']
                }
                
                analysis_data.append(analysis_record)
        
        # Save analysis dataset
        analysis_df = pd.DataFrame(analysis_data)
        analysis_path = f"{self.processed_dir}/fantasy_analysis_dataset.csv"
        analysis_df.to_csv(analysis_path, index=False)
        
        print(f"  ✅ {analysis_path} ({len(analysis_df)} records)")
        
        # Show summary statistics
        self.show_dataset_summary(analysis_df)
    
    def show_dataset_summary(self, df: pd.DataFrame):
        """Show summary statistics of the generated dataset"""
        print(f"\n📊 DATASET SUMMARY")
        print("=" * 40)
        print(f"Total records: {len(df):,}")
        print(f"Unique players: {df['player_id'].nunique():,}")
        print(f"Weeks covered: {df['week'].min()}-{df['week'].max()}")
        print(f"Positions: {list(df['position'].unique())}")
        
        print(f"\n🎯 PROJECTION PERFORMANCE:")
        beat_rate = df['beat_projection'].mean() * 100
        print(f"Players beating projections by 20%+: {beat_rate:.1f}%")
        
        avg_outperformance = df['outperformance_pct'].mean()
        print(f"Average outperformance: {avg_outperformance:+.1f}%")
        
        print(f"\n🏈 GAME CONTEXT:")
        print(f"High-scoring games (O/U > 48): {df['high_scoring_game'].mean()*100:.1f}%")
        print(f"Weather impact games: {df['weather_impact'].mean()*100:.1f}%")
        print(f"Home games: {df['is_home'].mean()*100:.1f}%")


def main():
    """Generate complete mock fantasy dataset"""
    generator = FantasyMockDataGenerator()
    
    # Generate 8 weeks of data (realistic sample size)
    dataset = generator.generate_full_dataset(weeks=8)
    
    # Save all datasets
    generator.save_datasets(dataset)
    
    print(f"\n🚀 MOCK DATA GENERATION COMPLETE!")
    print("=" * 50)
    print("✅ Ready to build prediction models!")
    print("✅ Ready to create web dashboard!")
    print("✅ Ready to demonstrate platform capabilities!")
    print("\n📁 Files created:")
    print("  • data/mock/ - Individual CSV files")
    print("  • data/processed/ - Master analysis dataset")
    print("=" * 50)


if __name__ == "__main__":
    main()