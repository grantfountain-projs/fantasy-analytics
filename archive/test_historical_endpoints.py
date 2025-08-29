"""
Test different ESPN historical endpoints to find working data sources
"""

import requests
import json
from pprint import pprint
from datetime import datetime, timedelta

class HistoricalEndpointTester:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
    
    def test_various_historical_formats(self):
        """Test different ways to request historical data"""
        
        # Different URL patterns to try for 2024 data
        test_urls = [
            # Basic 2024 scoreboard
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=2024",
            
            # Specific week in 2024
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=20240910",
            
            # Week-based approach
            "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2024/types/2/weeks/1",
            
            # Alternative format
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&week=1&year=2024",
            
            # Super Bowl 2024 (definitely completed)
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=20240211",
            
            # Regular season finale 2024
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=20241229",
        ]
        
        print("🔍 TESTING HISTORICAL ESPN ENDPOINTS")
        print("=" * 60)
        
        for i, url in enumerate(test_urls, 1):
            print(f"\n{i}. TESTING: {url}")
            print("-" * 50)
            
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                print(f"Status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Analyze the response
                        events = data.get('events', [])
                        print(f"✅ Found {len(events)} events")
                        
                        if events:
                            # Check first event
                            event = events[0]
                            print(f"Sample event: {event.get('name', 'Unknown')}")
                            print(f"Date: {event.get('date', 'Unknown')}")
                            status = event.get('status', {}).get('type', {}).get('name', 'Unknown')
                            print(f"Status: {status}")
                            
                            # Check for player stats
                            competitions = event.get('competitions', [])
                            if competitions:
                                comp = competitions[0]
                                competitors = comp.get('competitors', [])
                                
                                for j, competitor in enumerate(competitors[:2]):  # Check both teams
                                    team = competitor.get('team', {}).get('abbreviation', f'Team{j+1}')
                                    stats = competitor.get('statistics', [])
                                    leaders = competitor.get('leaders', [])
                                    
                                    print(f"  {team}: {len(stats)} stats, {len(leaders)} leader categories")
                                    
                                    if leaders:
                                        print(f"    Leader categories: {[l.get('name') for l in leaders[:3]]}")
                                        
                                        # Show sample player data
                                        for leader_cat in leaders[:1]:  # Just first category
                                            players = leader_cat.get('leaders', [])
                                            if players:
                                                player = players[0]
                                                athlete = player.get('athlete', {})
                                                print(f"    Sample player: {athlete.get('displayName')} - {player.get('displayValue')}")
                            
                            print(f"✅ SUCCESS - This endpoint has data!")
                        else:
                            print("⚠️  No events found")
                            
                    except json.JSONDecodeError:
                        print("❌ Invalid JSON response")
                        print(f"Response preview: {response.text[:200]}")
                        
                else:
                    print(f"❌ HTTP {response.status_code}")
                    print(f"Response: {response.text[:200]}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def test_specific_completed_game(self):
        """Test a specific game we know was completed"""
        print(f"\n{'='*60}")
        print("🏆 TESTING SPECIFIC COMPLETED GAME (Super Bowl 2024)")
        print("=" * 60)
        
        # Super Bowl LIV was February 11, 2024
        super_bowl_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=20240211"
        
        try:
            response = requests.get(super_bowl_url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                
                if events:
                    event = events[0]
                    print(f"Game: {event.get('name')}")
                    print(f"Status: {event.get('status', {}).get('type', {}).get('description')}")
                    
                    # Deep dive into this specific game
                    competitions = event.get('competitions', [])
                    if competitions:
                        comp = competitions[0]
                        competitors = comp.get('competitors', [])
                        
                        total_player_stats = 0
                        for competitor in competitors:
                            team = competitor.get('team', {}).get('displayName', 'Unknown')
                            leaders = competitor.get('leaders', [])
                            
                            print(f"\n{team}:")
                            for leader_cat in leaders:
                                category = leader_cat.get('name', 'Unknown')
                                players = leader_cat.get('leaders', [])
                                print(f"  {category}: {len(players)} players")
                                
                                total_player_stats += len(players)
                                
                                # Show top player in each category
                                if players:
                                    top_player = players[0]
                                    athlete = top_player.get('athlete', {})
                                    value = top_player.get('displayValue', 'N/A')
                                    print(f"    Top: {athlete.get('displayName', 'Unknown')} - {value}")
                        
                        print(f"\n🎯 TOTAL PLAYER STATS FOUND: {total_player_stats}")
                        
                        if total_player_stats > 0:
                            print("✅ SUCCESS! We can get historical player data!")
                            return True
                        else:
                            print("❌ No player stats found even in completed game")
                            return False
                    
                else:
                    print("❌ No events found for Super Bowl date")
                    return False
            else:
                print(f"❌ Failed to fetch Super Bowl data: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing Super Bowl: {e}")
            return False
    
    def suggest_alternative_approaches(self):
        """Suggest alternative data sources"""
        print(f"\n{'='*60}")
        print("💡 ALTERNATIVE APPROACHES")
        print("=" * 60)
        
        alternatives = [
            "1. Mock Data Generation: Create realistic fantasy data for testing",
            "2. Sports APIs: Try SportRadar, The Sports DB, or other sports APIs",
            "3. Web Scraping: Scrape ESPN's web pages instead of API",
            "4. Fantasy Platforms: Use Yahoo Fantasy or other APIs",
            "5. NFL Official API: Try NFL's official data sources"
        ]
        
        for alt in alternatives:
            print(f"   {alt}")
        
        print(f"\nFor now, let's create mock data to build the platform!")


def main():
    tester = HistoricalEndpointTester()
    
    # Test various endpoints
    tester.test_various_historical_formats()
    
    # Test a specific completed game
    success = tester.test_specific_completed_game()
    
    # If no real data found, suggest alternatives
    if not success:
        tester.suggest_alternative_approaches()
    
    print(f"\n{'='*60}")
    print("🎯 NEXT STEPS:")
    if success:
        print("✅ Found working historical data - update scraper!")
    else:
        print("⚠️  No historical data found - let's create mock data for demo")
    print("=" * 60)


if __name__ == "__main__":
    main()