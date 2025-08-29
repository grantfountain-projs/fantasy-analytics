"""
Historical ESPN Data Scraper

Gets completed game data from 2024 season for model training.
This will have actual player statistics we can use for fantasy analysis.
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import os

class HistoricalESPNScraper:
    def __init__(self):
        # ESPN API endpoints for historical data
        self.scoreboard_base = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        self.data_dir = "data/historical"
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Create data directories"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
    
    def get_historical_scoreboard(self, season: int = 2024, season_type: int = 2, week: int = 1) -> Dict:
        """
        Fetch historical scoreboard data for completed games
        
        Args:
            season: NFL season year (2024)
            season_type: 1=preseason, 2=regular, 3=postseason
            week: Week number
        """
        
        params = {
            'dates': f'{season}',
            'seasontype': season_type,
            'week': week
        }
        
        try:
            response = requests.get(self.scoreboard_base, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Fetched {season} Season, Week {week} data")
            return data
            
        except requests.RequestException as e:
            print(f"❌ Error fetching Week {week}: {e}")
            return {}
    
    def extract_player_stats_from_game(self, event: Dict) -> List[Dict]:
        """Extract player statistics from a completed game"""
        players = []
        
        try:
            game_id = event.get('id')
            game_name = event.get('shortName', 'Unknown')
            game_status = event.get('status', {}).get('type', {}).get('name', 'Unknown')
            
            # Only process completed games
            if 'FINAL' not in game_status.upper():
                print(f"  ⚠️ Skipping {game_name} - Status: {game_status}")
                return players
            
            print(f"  📊 Processing {game_name} - {game_status}")
            
            competitions = event.get('competitions', [])
            for competition in competitions:
                competitors = competition.get('competitors', [])
                
                for competitor in competitors:
                    team_info = competitor.get('team', {})
                    team_name = team_info.get('displayName', 'Unknown')
                    team_abbrev = team_info.get('abbreviation', 'UNK')
                    
                    # Extract team-level stats first
                    team_stats = competitor.get('statistics', [])
                    team_score = competitor.get('score', 0)
                    
                    # Look for individual player stats in leaders
                    leaders = competitor.get('leaders', [])
                    for leader_category in leaders:
                        category_name = leader_category.get('name', 'unknown')
                        category_leaders = leader_category.get('leaders', [])
                        
                        for player_stat in category_leaders:
                            athlete = player_stat.get('athlete', {})
                            
                            player = {
                                'game_id': game_id,
                                'game_name': game_name,
                                'game_status': game_status,
                                'player_id': athlete.get('id'),
                                'player_name': athlete.get('displayName', 'Unknown'),
                                'position': athlete.get('position', {}).get('abbreviation', 'UNK'),
                                'team_name': team_name,
                                'team_abbrev': team_abbrev,
                                'team_score': team_score,
                                'stat_category': category_name,
                                'stat_value': player_stat.get('value', 0),
                                'stat_display': player_stat.get('displayValue', '0'),
                                'jersey_number': athlete.get('jersey'),
                                'scraped_at': datetime.now().isoformat()
                            }
                            
                            if player['player_id']:
                                players.append(player)
            
            print(f"    ✅ Extracted {len(players)} player stats")
            
        except Exception as e:
            print(f"    ❌ Error processing game: {e}")
        
        return players
    
    def collect_week_data(self, season: int = 2024, week: int = 1) -> List[Dict]:
        """Collect all player data for a specific week"""
        print(f"\n🏈 Collecting {season} Season, Week {week}")
        print("-" * 50)
        
        # Get scoreboard data
        scoreboard_data = self.get_historical_scoreboard(season=season, week=week)
        
        if not scoreboard_data:
            return []
        
        # Process all games in the week
        all_players = []
        events = scoreboard_data.get('events', [])
        
        print(f"Found {len(events)} games for Week {week}")
        
        for event in events:
            game_players = self.extract_player_stats_from_game(event)
            all_players.extend(game_players)
        
        print(f"✅ Week {week} complete: {len(all_players)} total player stats")
        return all_players
    
    def save_week_data(self, players: List[Dict], season: int, week: int):
        """Save week data to CSV"""
        if not players:
            print(f"⚠️ No data to save for Week {week}")
            return
        
        df = pd.DataFrame(players)
        filename = f"{self.data_dir}/{season}_week_{week:02d}_players.csv"
        df.to_csv(filename, index=False)
        
        print(f"💾 Saved to {filename}")
        
        # Show summary
        print(f"📊 Week {week} Summary:")
        print(f"  • Total player stats: {len(df)}")
        print(f"  • Unique players: {df['player_id'].nunique()}")
        print(f"  • Stat categories: {df['stat_category'].unique()}")
        print(f"  • Teams: {df['team_abbrev'].unique()}")
    
    def collect_multiple_weeks(self, season: int = 2024, start_week: int = 1, end_week: int = 4):
        """Collect data for multiple weeks"""
        print(f"🚀 COLLECTING HISTORICAL DATA: {season} Season, Weeks {start_week}-{end_week}")
        print("=" * 70)
        
        all_data = []
        
        for week in range(start_week, end_week + 1):
            week_players = self.collect_week_data(season=season, week=week)
            
            if week_players:
                all_data.extend(week_players)
                self.save_week_data(week_players, season, week)
            
            # Be respectful to ESPN's servers
            time.sleep(2)
        
        # Create master dataset
        if all_data:
            self.create_master_dataset(all_data, season, start_week, end_week)
        
        print(f"\n🎯 COLLECTION COMPLETE!")
        print(f"Total player stats collected: {len(all_data)}")
    
    def create_master_dataset(self, all_data: List[Dict], season: int, start_week: int, end_week: int):
        """Create combined dataset from all weeks"""
        df = pd.DataFrame(all_data)
        filename = f"data/processed/historical_{season}_weeks_{start_week}-{end_week}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n📊 MASTER DATASET: {filename}")
        print(f"  • Total records: {len(df)}")
        print(f"  • Unique players: {df['player_id'].nunique()}")
        print(f"  • Date range: Weeks {start_week}-{end_week}")
        print(f"  • Stat categories: {list(df['stat_category'].unique())}")
        
        # Show top performers
        if len(df) > 0:
            print(f"\n🌟 TOP PERFORMERS:")
            top_players = df.groupby(['player_name', 'stat_category'])['stat_value'].sum().sort_values(ascending=False).head(10)
            for (player, category), value in top_players.items():
                print(f"  • {player} ({category}): {value}")


def main():
    """Main execution function"""
    print("🏈 ESPN HISTORICAL DATA SCRAPER")
    print("=" * 50)
    print("Getting completed games from 2024 season for model training")
    print("=" * 50)
    
    scraper = HistoricalESPNScraper()
    
    # Collect first 4 weeks of 2024 season (should have completed games)
    scraper.collect_multiple_weeks(season=2024, start_week=1, end_week=4)
    
    print("\n" + "=" * 50)
    print("✅ Ready for analysis!")
    print("Next steps:")
    print("1. Examine the historical data CSV files")
    print("2. Build prediction models")
    print("3. Test against known outcomes")
    print("=" * 50)


if __name__ == "__main__":
    main()