import json
import requests
from datetime import datetime, timedelta
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # magic: add project root directory to path
from src.keys import news_api_key
# Your News API key
API_KEY = news_api_key


def fetch_latest_news():
    # Base URL for the News API
    url = "https://newsapi.org/v2/everything"

    # Generate date range from Jan 1, 2025 to today
    start_date = datetime(2025, 1, 31)
    end_date = datetime.now()
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Parameters for the API request
        params = {
            'q': '*',  # Fetch all topics
            'from': date_str,
            'to': date_str,  # Same as from date to get single day
            'sortBy': 'popularity',  
            'pageSize': 50,
            'language': 'en',
            'apiKey': API_KEY
        }

        try:
            # Make the request to News API
            response = requests.get(url, params=params)
            response.raise_for_status()

            # Parse the JSON response
            articles = response.json().get('articles', [])
            
            # Write to JSONL file
            with open('real_recent_news.jsonl', 'a', encoding='utf-8') as f:
                for article in articles:
                    article_data = {
                        'title': article['title'],
                        'description': article['description'],
                        'content': article['content'],
                        'published_at': article['publishedAt']
                    }
                    json.dump(article_data, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"Successfully wrote {len(articles)} articles for {date_str}")
            
            # Add delay to respect API rate limits
            time.sleep(0.1)  # Add 1 second delay between requests
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news for {date_str}: {e}")
        
        # Move to next day
        current_date = current_date + timedelta(days=1)

if __name__ == "__main__":
    fetch_latest_news()
