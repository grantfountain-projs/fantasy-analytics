"""
Detailed ESPN Data Explorer

Deep dive into the actual structure to find player data
"""

import json
import os
from pprint import pprint

class DetailedExplorer:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
    
    def get_latest_scoreboard(self):
        """Get the latest scoreboard file"""
        files = [f for f in os.listdir(self.data_dir) if f.startswith('scoreboard_')]
        return sorted(files)[-1] if files else None
    
    def explore_events_structure(self):
        """Deep dive into events structure where player data likely exists"""
        scoreboard_file = self.get_latest_scoreboard()
        if not scoreboard_file:
            print("❌ No scoreboard file found")
            return
            
        filepath = f"{self.data_dir}/{scoreboard_file}"
        print(f"🔍 DEEP DIVE: {filepath}")
        print("=" * 60)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        events = data.get('events', [])
        print(f"📊 Found {len(events)} events (games)")
        
        if not events:
            print("❌ No events found")
            return
        
        # Examine first event in detail
        event = events[0]
        print(f"\n🏈 EXAMINING EVENT: {event.get('name', 'Unknown Game')}")
        print("-" * 50)
        
        print("Event top-level keys:")
        for key in event.keys():
            value = event[key]
            if isinstance(value, list):
                print(f"  • {key}: list ({len(value)} items)")
            elif isinstance(value, dict):
                print(f"  • {key}: dict ({len(value)} keys)")
            else:
                print(f"  • {key}: {type(value).__name__} = {str(value)[:50]}")
        
        # Look for competitions (where team/player data usually is)
        if 'competitions' in event:
            competitions = event['competitions']
            print(f"\n🏆 Found {len(competitions)} competition(s)")
            
            if competitions:
                comp = competitions[0]
                print("\nCompetition keys:")
                for key in comp.keys():
                    value = comp[key]
                    if isinstance(value, list):
                        print(f"  • {key}: list ({len(value)} items)")
                    elif isinstance(value, dict):
                        print(f"  • {key}: dict ({len(value)} keys)")
                    else:
                        print(f"  • {key}: {type(value).__name__}")
                
                # Examine competitors (teams)
                if 'competitors' in comp:
                    competitors = comp['competitors']
                    print(f"\n👥 Found {len(competitors)} competitor(s) (teams)")
                    
                    for i, competitor in enumerate(competitors):
                        team = competitor.get('team', {})
                        team_name = team.get('displayName', 'Unknown')
                        print(f"\n  TEAM {i+1}: {team_name}")
                        print(f"  Competitor keys: {list(competitor.keys())}")
                        
                        # Look for player data
                        if 'roster' in competitor:
                            roster = competitor['roster']
                            print(f"    📋 ROSTER: {type(roster)} with {len(roster) if hasattr(roster, '__len__') else 'unknown'} items")
                        
                        if 'athletes' in competitor:
                            athletes = competitor['athletes']
                            print(f"    🏃 ATHLETES: {type(athletes)} with {len(athletes) if hasattr(athletes, '__len__') else 'unknown'} items")
                        
                        if 'statistics' in competitor:
                            stats = competitor['statistics']
                            print(f"    📊 STATISTICS: {type(stats)} with {len(stats) if hasattr(stats, '__len__') else 'unknown'} items")
                            
                            # Show sample stats
                            if isinstance(stats, list) and stats:
                                print("    Sample statistics:")
                                pprint(stats[:2], depth=2, width=80)
        
        # Also check if there are leaders/statistics at event level
        if 'leaders' in event:
            leaders = event['leaders']
            print(f"\n🌟 EVENT LEADERS: {type(leaders)}")
            if isinstance(leaders, list):
                for leader in leaders[:2]:  # Show first 2
                    print(f"  Leader type: {leader.get('name', 'Unknown')}")
                    if 'leaders' in leader:
                        print(f"    Players: {len(leader['leaders'])}")
    
    def explore_fantasy_structure(self):
        """Explore fantasy games JSON structure"""
        files = [f for f in os.listdir(self.data_dir) if f.startswith('fantasy_')]
        fantasy_file = sorted(files)[-1] if files else None
        
        if not fantasy_file:
            print("❌ No fantasy file found")
            return
        
        filepath = f"{self.data_dir}/{fantasy_file}"
        print(f"\n🏆 FANTASY DATA: {filepath}")
        print("=" * 60)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        events = data.get('events', [])
        print(f"📊 Found {len(events)} fantasy events")
        
        if events:
            event = events[0]
            print(f"\nSample fantasy event keys: {list(event.keys())}")
            
            # Look for competitor data
            if 'competitors' in event:
                competitors = event['competitors']
                print(f"Fantasy competitors: {len(competitors)}")
                if competitors:
                    comp = competitors[0]
                    print(f"Competitor keys: {list(comp.keys())}")
    
    def look_for_player_stats(self):
        """Specifically hunt for actual player statistics"""
        scoreboard_file = self.get_latest_scoreboard()
        if not scoreboard_file:
            return
            
        filepath = f"{self.data_dir}/{scoreboard_file}"
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"\n🎯 HUNTING FOR PLAYER STATS")
        print("=" * 50)
        
        # Check if this is preseason (no stats yet)
        season_info = data.get('season', {})
        print(f"Season type: {season_info}")
        
        events = data.get('events', [])
        for i, event in enumerate(events[:3]):  # Check first 3 games
            print(f"\n🏈 Game {i+1}: {event.get('shortName', 'Unknown')}")
            status = event.get('status', {}).get('type', {}).get('name', 'Unknown')
            print(f"Status: {status}")
            
            # Games might not have started yet, which explains no player stats
            if status in ['STATUS_SCHEDULED', 'STATUS_POSTPONED']:
                print("  ⚠️  Game hasn't started - no player stats available yet")
                continue
            
            # Look deeper in competitions
            competitions = event.get('competitions', [])
            for comp in competitions:
                competitors = comp.get('competitors', [])
                for competitor in competitors:
                    team_name = competitor.get('team', {}).get('abbreviation', 'UNK')
                    
                    # Check all possible stat locations
                    stat_locations = ['statistics', 'leaders', 'roster', 'athletes']
                    for location in stat_locations:
                        if location in competitor:
                            data_found = competitor[location]
                            if data_found:
                                print(f"  ✅ {team_name} - Found {location}: {type(data_found)} ({len(data_found) if hasattr(data_found, '__len__') else 'N/A'})")
                            else:
                                print(f"  ❌ {team_name} - Empty {location}")


def main():
    print("🔍 DETAILED ESPN DATA EXPLORATION")
    print("=" * 60)
    
    explorer = DetailedExplorer()
    
    # Deep dive into scoreboard structure
    explorer.explore_events_structure()
    
    # Check fantasy structure
    explorer.explore_fantasy_structure()
    
    # Hunt specifically for player stats
    explorer.look_for_player_stats()
    
    print("\n" + "=" * 60)
    print("🎯 ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()