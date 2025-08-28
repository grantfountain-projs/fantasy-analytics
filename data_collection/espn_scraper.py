"""
ESPN Fantasy Football Data Scraper

Collects player projections and actual performance data from ESPN Fantasy API
for analysis and model training.
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

class ESPNFantasyScraper:
    def __init__(self):
        self.base_url = "https://fantasy.espn.com/apis/v3/games/ffl/seasons/2024"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.data_dir = "data/raw"
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
    
    def get_current_week(self) -> int:
        """Determine current NFL week"""
        # Simple estimation - in production, you'd use NFL schedule API
        season_start = datetime(2024, 9, 5)  # Approximate 2024 season start
        current_date = datetime.now()
        days_into_season = (current_date - season_start).days
        week = max(1, min(18, (days_into_season // 7) + 1))
        return week
    
    def fetch_player_stats(self, week: int, season: int = 2024) -> List[Dict]:
        """
        Fetch player statistics for a specific week
        
        Args:
            week: NFL week number (1-18)
            season: NFL season year
            
        Returns:
            List of player stat dictionaries
        """
        url = f"{self.base_url}/segments/0/leagues/0"
        
        params = {
            'view': ['mMatchup', 'mMatchupScore'],
            'scoringPeriodId': week
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            players = self.parse_player_data(data, week)
            
            print(f"Successfully fetched {len(players)} players for Week {week}")
            return players
            
        except requests.RequestException as e:
            print(f"Error fetching data for Week {week}: {e}")
            return []
    
    def parse_player_data(self, raw_data: Dict, week: int) -> List[Dict]:
        """
        Parse raw ESPN API response into structured player data
        
        Args:
            raw_data: Raw API response
            week: NFL week number
            
        Returns:
            List of structured player dictionaries
        """
        players = []
        
        # ESPN API structure can be complex - this is a simplified version
        # In practice, you'd need to handle the specific structure of their response
        
        try:
            # This is a placeholder structure - actual ESPN API parsing would be more complex
            if 'teams' in raw_data:
                for team in raw_data.get('teams', []):
                    roster = team.get('roster', {})
                    for entry in roster.get('entries', []):
                        player_info = entry.get('playerPoolEntry', {}).get('player', {})
                        
                        player = {
                            'player_id': player_info.get('id'),
                            'name': player_info.get('fullName', 'Unknown'),
                            'position': self.get_position(player_info),
                            'team': self.get_team(player_info),
                            'week': week,
                            'season': 2024,
                            'projected_points': self.get_projected_points(player_info),
                            'actual_points': self.get_actual_points(entry, week),
                            'scraped_at': datetime.now().isoformat()
                        }
                        
                        if player['player_id']:
                            players.append(player)
                            
        except KeyError as e:
            print(f"Error parsing player data: {e}")
            
        return players
    
    def get_position(self, player_info: Dict) -> str:
        """Extract player position from player info"""
        default_pos = player_info.get('defaultPositionId', 0)
        position_map = {1: 'QB', 2: 'RB', 3: 'WR', 4: 'TE', 5: 'K', 16: 'DST'}
        return position_map.get(default_pos, 'UNKNOWN')
    
    def get_team(self, player_info: Dict) -> str:
        """Extract player team from player info"""
        pro_team = player_info.get('proTeamId', 0)
        # Would need full team mapping - simplified for now
        return f"TEAM_{pro_team}"
    
    def get_projected_points(self, player_info: Dict) -> float:
        """Extract projected points from player info"""
        # Placeholder - actual implementation would parse ESPN's projection format
        return player_info.get('projected_points', 0.0)
    
    def get_actual_points(self, entry: Dict, week: int) -> Optional[float]:
        """Extract actual points scored"""
        # Placeholder - would parse actual scoring data
        return entry.get('actual_points')
    
    def save_weekly_data(self, players: List[Dict], week: int):
        """Save player data to CSV file"""
        if not players:
            print(f"No data to save for Week {week}")
            return
            
        df = pd.DataFrame(players)
        filename = f"{self.data_dir}/week_{week}_players.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {len(players)} players to {filename}")
    
    def collect_historical_data(self, start_week: int = 1, end_week: int = None):
        """
        Collect data for multiple weeks
        
        Args:
            start_week: Starting week number
            end_week: Ending week number (defaults to current week)
        """
        if end_week is None:
            end_week = self.get_current_week()
            
        print(f"Collecting data for Weeks {start_week} to {end_week}")
        
        for week in range(start_week, end_week + 1):
            print(f"\nFetching Week {week} data...")
            players = self.fetch_player_stats(week)
            self.save_weekly_data(players, week)
            
            # Be respectful with API calls
            time.sleep(2)
            
        print(f"\nData collection complete!")
    
    def create_master_dataset(self):
        """Combine all weekly data into a master dataset"""
        all_files = [f for f in os.listdir(self.data_dir) if f.startswith('week_') and f.endswith('.csv')]
        
        if not all_files:
            print("No weekly data files found")
            return
            
        dfs = []
        for file in sorted(all_files):
            df = pd.read_csv(f"{self.data_dir}/{file}")
            dfs.append(df)
            
        master_df = pd.concat(dfs, ignore_index=True)
        master_df.to_csv("data/processed/master_dataset.csv", index=False)
        
        print(f"Created master dataset with {len(master_df)} total player records")
        print(f"Weeks covered: {sorted(master_df['week'].unique())}")


def main():
    """Main execution function"""
    scraper = ESPNFantasyScraper()
    
    # Start with current week for testing
    current_week = scraper.get_current_week()
    print(f"Current NFL Week: {current_week}")
    
    # Collect data for current week
    scraper.collect_historical_data(start_week=current_week, end_week=current_week)
    
    # Uncomment below to collect more historical data
    # scraper.collect_historical_data(start_week=1, end_week=current_week)
    
    # Create master dataset
    scraper.create_master_dataset()


if __name__ == "__main__":
    main()