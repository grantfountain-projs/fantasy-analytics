"""
Debug script to understand ESPN's API response
"""

import requests
import json
from pprint import pprint

def test_espn_endpoints():
    """Test different ESPN endpoints to find working ones"""
    
    # Different URL patterns to try
    endpoints = [
        "https://fantasy.espn.com/apis/v3/games/ffl/seasons/2024/segments/0/leagues/0",
        "https://fantasy.espn.com/apis/v3/games/ffl/seasons/2024",
        "https://site.api.espn.com/apis/fantasy/v2/games/ffl/games",
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
        "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2024/types/2/weeks/18"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    for i, url in enumerate(endpoints, 1):
        print(f"\n{'='*80}")
        print(f"TESTING ENDPOINT {i}: {url}")
        print('='*80)
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"Status Code: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")
            print(f"Response Length: {len(response.text)} characters")
            
            # Show first 500 characters of response
            print(f"\nFirst 500 characters:")
            print("-" * 40)
            print(response.text[:500])
            print("-" * 40)
            
            # Try to parse as JSON
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"\n✅ Valid JSON! Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
                    # If it's a dict, show structure
                    if isinstance(data, dict):
                        print("\nJSON Structure:")
                        for key, value in list(data.items())[:5]:  # First 5 keys
                            value_type = type(value).__name__
                            if isinstance(value, (list, dict)):
                                length = len(value) if hasattr(value, '__len__') else 'Unknown'
                                print(f"  {key}: {value_type} (length: {length})")
                            else:
                                print(f"  {key}: {value_type} = {str(value)[:50]}")
                                
                except json.JSONDecodeError as e:
                    print(f"❌ JSON Parse Error: {e}")
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except requests.RequestException as e:
            print(f"❌ Request Error: {e}")
            
        print("\n" + "="*80)

def test_specific_fantasy_league():
    """Test with a known public fantasy league if possible"""
    
    # Try ESPN's demo/public endpoints
    demo_urls = [
        "https://fantasy.espn.com/apis/v3/games/ffl/seasons/2024/segments/0/leagues/252353",
        "https://fantasy.espn.com/apis/v3/games/ffl/seasons/2024?view=mMatchup",
        "https://fantasy.espn.com/apis/v3/games/ffl/news",
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    
    print("\nTesting Fantasy-Specific Endpoints:")
    print("="*50)
    
    for url in demo_urls:
        print(f"\nTrying: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=5)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print(f"Success! Response length: {len(response.text)}")
                try:
                    data = response.json()
                    print(f"JSON keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                except:
                    print("Not valid JSON")
            else:
                print(f"Response: {response.text[:200]}")
        except Exception as e:
            print(f"Error: {e}")

def main():
    print("ESPN Fantasy API Debug Tool")
    print("="*50)
    
    # Test general endpoints
    test_espn_endpoints()
    
    # Test fantasy-specific endpoints
    test_specific_fantasy_league()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("Look for endpoints that return:")
    print("1. Status Code: 200")
    print("2. Valid JSON")
    print("3. Player/fantasy-related data")
    print("="*50)

if __name__ == "__main__":
    main()