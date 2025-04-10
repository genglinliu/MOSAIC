import logging
import random
import sqlite3
import jsonlines
from utils import Utils
from agent_user import AgentUser

class NewsManager:
    def __init__(self, config: dict, conn: sqlite3.Connection):
        self.config = config
        self.conn = conn
        self.news_agent = self._create_news_agent()
        # Add tracking for used articles
        self.used_real_articles = set()
        self.used_fake_articles = set()
        # Track indices for chronological diffusion of both real and fake news
        self.next_real_news_index = 0
        self.next_fake_news_index = 0
    
    def _create_news_agent(self) -> AgentUser:
        """Create a specialized news agent."""
        news_config = {
            'persona': 'Professional news organization focused on delivering accurate and timely news updates.',
        }
        
        news_agent_id = "agentverse_news"
        
        # Register news agent in database
        self.conn.execute('''
            INSERT INTO users (
                user_id, persona, creation_time
            )
            VALUES (?, ?, datetime('now'))
        ''', (news_agent_id, news_config['persona']))
        self.conn.commit()
        
        return AgentUser(
            user_id=news_agent_id,
            user_config=news_config,
            temperature=0.3,  # Lower temperature for more consistent news reporting
            is_news_agent=True
        )

    def inject_news(self) -> list[str]:
        """Inject multiple news articles through the specialized news agent.
        
        Returns:
            list[str]: List of post IDs for the injected news articles
        """
        num_articles = self.config['news_injection'].get('articles_per_injection', 1)
        post_ids = []
        
        for i in range(num_articles):
            # Use a 9:1 ratio of real to fake news (10% fake news)
            # Every 10th article will be fake news
            news_type = 'fake' if i % 10 == 9 else 'real'
            news_file = 'data/real_recent_news.jsonl' if news_type == 'real' else 'data/fake_news.jsonl'
            
            # Load all articles
            with jsonlines.open(news_file) as reader:
                all_articles = list(reader)
            
            if news_type == 'real':
                # For real news, select articles in chronological order
                if self.next_real_news_index >= len(all_articles):
                    # Reset if we've used all articles
                    self.next_real_news_index = 0
                
                selected_article = all_articles[self.next_real_news_index]
                self.next_real_news_index += 1
                content = f"[NEWS] {selected_article['title']}: {selected_article['description']} {selected_article['content']}"
            else:
                # For fake news, also use chronological order
                if self.next_fake_news_index >= len(all_articles):
                    # Reset if we've used all articles
                    self.next_fake_news_index = 0
                
                selected_article = all_articles[self.next_fake_news_index]
                self.next_fake_news_index += 1
                content = f"[NEWS] {selected_article['name']}: {selected_article['description']}"
                
            post_id = self.news_agent.create_post(content, is_news=True, news_type=news_type, status='active')
            
            post_ids.append(post_id)
            
        self.conn.commit()
        logging.info(f"Injected {num_articles} news articles through user {self.news_agent.user_id}")
        
        return post_ids

    def get_news_stats(self, news_post_id: str) -> dict:
        """Get statistics for a specific news post."""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT 
                p.num_likes, 
                p.num_shares, 
                p.num_comments, 
                p.num_flags,
                COUNT(cn.note_id) as total_notes,
                SUM(cn.helpful_ratings) as total_helpful_ratings
            FROM posts p
            LEFT JOIN community_notes cn ON p.post_id = cn.post_id
            WHERE p.post_id = ?
            GROUP BY p.post_id
        ''', (news_post_id,))
        
        stats = cursor.fetchone()
        
        if stats:
            return {
                'likes': stats[0],
                'shares': stats[1],
                'comments': stats[2],
                'flags': stats[3],
                'notes': stats[4] or 0,
                'helpful_ratings': stats[5] or 0
            }
        return None