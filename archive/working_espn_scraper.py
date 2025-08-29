"""
Working ESPN NFL Data Scraper

Uses ESPN's public APIs that actually return JSON data.
Focuses on player performance and game data for fantasy analysis.
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import os

class WorkingESPNScraper:
    def __init__(self):
        # Working ESPN API endpoints
        self.fantasy_games_url = "https://site.api.espn.com/apis/fantasy/v2/games/ffl/games"
        self.scoreboard_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        self.player_stats_base = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/athletes"
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        self.data_dir = "data/raw"
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Create data directories if they don't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
    
    def fetch_current_games(self) -> Dict:
        """Fetch current NFL games and scores"""
        try:
            response = requests.get(self.scoreboard_url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Successfully fetched NFL scoreboard data")
            return data
            
        except requests.RequestException as e:
            print(f"❌ Error fetching scoreboard: {e}")
            return {}
    
    def fetch_fantasy_games(self) -> Dict:
        """Fetch fantasy football games data"""
        try:
            response = requests.get(self.fantasy_games_url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Successfully fetched fantasy games data")
            return data
            
        except requests.RequestException as e:
            print(f"❌ Error fetching fantasy games: {e}")
            return {}
    
    def parse_player_performances(self, scoreboard_data: Dict) -> List[Dict]:
        """Extract player performance data from scoreboard"""
        players = []
        
        try:
            events = scoreboard_data.get('events', [])
            
            for event in events:
                # Extract game info
                game_id = event.get('id')
                game_status = event.get('status', {}).get('type', {}).get('name', 'Unknown')
                
                # Get teams
                competitions = event.get('competitions', [])
                for competition in competitions:
                    competitors = competition.get('competitors', [])
                    
                    for team_data in competitors:
                        team = team_data.get('team', {})
                        team_name = team.get('displayName', 'Unknown Team')
                        team_abbrev = team.get('abbreviation', 'UNK')
                        
                        # Extract player stats if available
                        if 'statistics' in competition:
                            stats = competition.get('statistics', [])
                            for stat_category in stats:
                                athletes = stat_category.get('athletes', [])
                                for athlete in athletes:
                                    player = self.extract_player_info(
                                        athlete, game_id, team_name, team_abbrev, game_status
                                    )
                                    if player:
                                        players.append(player)
        
        except Exception as e:
            print(f"⚠️  Error parsing player performances: {e}")
        
        return players
    
    def extract_player_info(self, athlete_data: Dict, game_id: str, 
                           team_name: str, team_abbrev: str, game_status: str) -> Optional[Dict]:
        """Extract individual player information"""
        try:
            athlete = athlete_data.get('athlete', {})
            
            player = {
                'player_id': athlete.get('id'),
                'name': athlete.get('displayName', 'Unknown'),
                'position': athlete.get('position', {}).get('abbreviation', 'UNK'),
                'team_name': team_name,
                'team_abbrev': team_abbrev,
                'game_id': game_id,
                'game_status': game_status,
                'jersey_number': athlete.get('jersey'),
                'scraped_at': datetime.now().isoformat()
            }
            
            # Extract statistics
            stats = athlete_data.get('statistics', [])
            for stat in stats:
                stat_name = stat.get('name', '').lower().replace(' ', '_')
                stat_value = stat.get('displayValue', stat.get('value', 0))
                player[f'stat_{stat_name}'] = stat_value
            
            return player if player['player_id'] else None
            
        except Exception as e:
            print(f"⚠️  Error extracting player {athlete_data}: {e}")
            return None
    
    def get_week_number(self) -> int:
        """Get current NFL week from scoreboard data"""
        try:
            scoreboard = self.fetch_current_games()
            week_info = scoreboard.get('week', {})
            return week_info.get('number', 1)
        except:
            # Fallback calculation
            season_start = datetime(2024, 9, 5)
            current_date = datetime.now()
            days_into_season = (current_date - season_start).days
            return max(1, min(18, (days_into_season // 7) + 1))
    
    def collect_current_data(self):
        """Collect all current data from working ESPN APIs"""
        print("🏈 Starting ESPN data collection...")
        
        # Get scoreboard data
        print("\n📊 Fetching NFL scoreboard...")
        scoreboard_data = self.fetch_current_games()
        
        # Get fantasy games data  
        print("🏆 Fetching fantasy games...")
        fantasy_data = self.fetch_fantasy_games()
        
        # Parse player data
        print("👥 Parsing player performances...")
        players = self.parse_player_performances(scoreboard_data)
        
        # Get current week
        current_week = self.get_week_number()
        print(f"📅 Current NFL Week: {current_week}")
        
        # Save raw data
        self.save_raw_data(scoreboard_data, fantasy_data, current_week)
        
        # Save parsed player data
        self.save_player_data(players, current_week)
        
        return {
            'players': players,
            'week': current_week,
            'games_count': len(scoreboard_data.get('events', [])),
            'fantasy_events': len(fantasy_data.get('events', []))
        }
    
    def save_raw_data(self, scoreboard: Dict, fantasy: Dict, week: int):
        """Save raw API responses"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save scoreboard data
        scoreboard_file = f"{self.data_dir}/scoreboard_week_{week}_{timestamp}.json"
        with open(scoreboard_file, 'w') as f:
            json.dump(scoreboard, f, indent=2)
        print(f"💾 Saved scoreboard data to {scoreboard_file}")
        
        # Save fantasy data
        fantasy_file = f"{self.data_dir}/fantasy_week_{week}_{timestamp}.json"
        with open(fantasy_file, 'w') as f:
            json.dump(fantasy, f, indent=2)
        print(f"💾 Saved fantasy data to {fantasy_file}")
    
    def save_player_data(self, players: List[Dict], week: int):
        """Save parsed player data to CSV"""
        if not players:
            print("⚠️  No player data to save")
            return
            
        df = pd.DataFrame(players)
        filename = f"{self.data_dir}/players_week_{week}.csv"
        df.to_csv(filename, index=False)
        
        print(f"💾 Saved {len(players)} players to {filename}")
        
        # Show sample of what we collected
        print("\n📋 Sample of collected data:")
        print(df[['name', 'position', 'team_abbrev', 'game_status']].head())
    
    def analyze_data_quality(self, result: Dict):
        """Analyze the quality of collected data"""
        print("\n📈 Data Quality Analysis:")
        print(f"  • Players collected: {len(result.get('players', []))}")
        print(f"  • Games found: {result.get('games_count', 0)}")
        print(f"  • Fantasy events: {result.get('fantasy_events', 0)}")
        print(f"  • Current week: {result.get('week', 'Unknown')}")
        
        players = result.get('players', [])
        if players:
            df = pd.DataFrame(players)
            positions = df['position'].value_counts()
            print(f"  • Positions found: {dict(positions)}")
            
            # Check for actual stats
            stat_columns = [col for col in df.columns if col.startswith('stat_')]
            print(f"  • Stat categories: {len(stat_columns)}")
            if stat_columns:
                print(f"  • Stats available: {stat_columns[:5]}...")  # Show first 5


def main():
    """Main execution function"""
    print("🚀 ESPN NFL Data Scraper (Working Version)")
    print("=" * 50)
    
    scraper = WorkingESPNScraper()
    
    # Collect current data
    result = scraper.collect_current_data()
    
    # Analyze what we got
    scraper.analyze_data_quality(result)
    
    print("\n" + "=" * 50)
    print("✅ Data collection complete!")
    print("\nNext steps:")
    print("1. Check the CSV files in data/raw/")
    print("2. Analyze the data structure")
    print("3. Build prediction models")
    print("=" * 50)


if __name__ == "__main__":
    main()