"""
JSON Data Explorer

Explores the structure of ESPN API responses to understand 
where player data is actually located.
"""

import json
import os
from pprint import pprint
from typing import Dict, Any

class JSONExplorer:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
    
    def get_latest_files(self):
        """Get the most recent JSON files"""
        files = os.listdir(self.data_dir)
        
        scoreboard_files = [f for f in files if f.startswith('scoreboard_')]
        fantasy_files = [f for f in files if f.startswith('fantasy_')]
        
        latest_scoreboard = sorted(scoreboard_files)[-1] if scoreboard_files else None
        latest_fantasy = sorted(fantasy_files)[-1] if fantasy_files else None
        
        return latest_scoreboard, latest_fantasy
    
    def explore_json_structure(self, filepath: str, max_depth: int = 3):
        """Explore JSON structure recursively"""
        print(f"\n{'='*60}")
        print(f"EXPLORING: {filepath}")
        print('='*60)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.print_structure(data, depth=0, max_depth=max_depth)
    
    def print_structure(self, obj: Any, depth: int = 0, max_depth: int = 3, key: str = "root"):
        """Recursively print JSON structure"""
        indent = "  " * depth
        
        if depth > max_depth:
            print(f"{indent}... (max depth reached)")
            return
        
        if isinstance(obj, dict):
            print(f"{indent}{key}: dict ({len(obj)} keys)")
            for k, v in list(obj.items())[:5]:  # Show first 5 keys
                self.print_structure(v, depth + 1, max_depth, k)
            if len(obj) > 5:
                print(f"{indent}  ... and {len(obj) - 5} more keys")
                
        elif isinstance(obj, list):
            print(f"{indent}{key}: list ({len(obj)} items)")
            if obj:  # If list is not empty
                print(f"{indent}  [0]: {type(obj[0]).__name__}")
                if len(obj) > 0:
                    self.print_structure(obj[0], depth + 1, max_depth, "[0]")
                if len(obj) > 1:
                    print(f"{indent}  ... and {len(obj) - 1} more items")
            else:
                print(f"{indent}  (empty list)")
                
        else:
            value_str = str(obj)[:50] + "..." if len(str(obj)) > 50 else str(obj)
            print(f"{indent}{key}: {type(obj).__name__} = {value_str}")
    
    def search_for_players(self, data: Dict, path: str = ""):
        """Search for player-related data in JSON structure"""
        player_data = []
        
        def recursive_search(obj, current_path):
            if isinstance(obj, dict):
                # Look for player-related keys
                player_keys = ['athletes', 'players', 'roster', 'statistics', 'competitors']
                
                for key, value in obj.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    
                    if key.lower() in [k.lower() for k in player_keys]:
                        player_data.append({
                            'path': new_path,
                            'key': key,
                            'type': type(value).__name__,
                            'length': len(value) if hasattr(value, '__len__') else 'N/A'
                        })
                    
                    recursive_search(value, new_path)
                    
            elif isinstance(obj, list) and obj:
                # Check first item in list
                recursive_search(obj[0], f"{current_path}[0]")
        
        recursive_search(data, path)
        return player_data
    
    def examine_specific_paths(self, filepath: str, paths: list):
        """Examine specific paths in the JSON that might contain player data"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"\n🔍 EXAMINING SPECIFIC PATHS in {filepath}")
        print("=" * 60)
        
        for path in paths:
            print(f"\nPath: {path}")
            print("-" * 40)
            
            try:
                # Navigate to the path
                current = data
                for step in path.split('.'):
                    if step.startswith('[') and step.endswith(']'):
                        # Array index
                        index = int(step[1:-1])
                        current = current[index]
                    else:
                        current = current[step]
                
                # Print what we found
                if isinstance(current, list):
                    print(f"Found list with {len(current)} items")
                    if current:
                        print("First item:")
                        pprint(current[0], depth=2, width=80)
                elif isinstance(current, dict):
                    print(f"Found dict with keys: {list(current.keys())}")
                    pprint(current, depth=1, width=80)
                else:
                    print(f"Found: {current}")
                    
            except (KeyError, IndexError, TypeError) as e:
                print(f"Path not found: {e}")
    
    def full_exploration(self):
        """Complete exploration of both JSON files"""
        scoreboard_file, fantasy_file = self.get_latest_files()
        
        if not scoreboard_file and not fantasy_file:
            print("❌ No JSON files found! Run the scraper first.")
            return
        
        # Explore scoreboard data
        if scoreboard_file:
            scoreboard_path = f"{self.data_dir}/{scoreboard_file}"
            self.explore_json_structure(scoreboard_path)
            
            # Search for player data
            with open(scoreboard_path, 'r') as f:
                scoreboard_data = json.load(f)
            
            player_paths = self.search_for_players(scoreboard_data)
            if player_paths:
                print(f"\n🎯 FOUND POTENTIAL PLAYER DATA PATHS:")
                for item in player_paths:
                    print(f"  • {item['path']} ({item['type']}, length: {item['length']})")
                
                # Examine promising paths
                promising_paths = [item['path'] for item in player_paths if item['length'] != 'N/A' and item['length'] > 0]
                if promising_paths:
                    self.examine_specific_paths(scoreboard_path, promising_paths[:3])
            
        # Explore fantasy data
        if fantasy_file:
            fantasy_path = f"{self.data_dir}/{fantasy_file}"
            self.explore_json_structure(fantasy_path)
            
            # Search for player data
            with open(fantasy_path, 'r') as f:
                fantasy_data = json.load(f)
            
            player_paths = self.search_for_players(fantasy_data)
            if player_paths:
                print(f"\n🎯 FOUND POTENTIAL PLAYER DATA PATHS:")
                for item in player_paths:
                    print(f"  • {item['path']} ({item['type']}, length: {item['length']})")


def main():
    """Main exploration function"""
    print("🔍 ESPN JSON Data Explorer")
    print("=" * 50)
    
    explorer = JSONExplorer()
    explorer.full_exploration()
    
    print("\n" + "=" * 50)
    print("✅ Exploration complete!")
    print("\nUse this information to update the scraper's")
    print("parsing logic to extract player data correctly.")
    print("=" * 50)


if __name__ == "__main__":
    main()