import logging
import sqlite3
import json
from typing import List, Literal
from pydantic import BaseModel
from openai import OpenAI
from post import Post, CommunityNote
from utils import Utils
from prompts import FactCheckerPrompts
from keys import OPENAI_API_KEY


class FactCheckVerdict(BaseModel):
    """
    A fact-check verdict for a post.
    This class is used to validate and structure the output from the LLM.
    """
    verdict: Literal["true", "false", "unverified"]
    explanation: str
    confidence: float  # 0.0 to 1.0
    sources: List[str]  # References from LLM's training data (note: not real-time sources)

class FactChecker:
    """
    A fact-checking agent that verifies content and provides verdicts.
    
    This agent acts as a platform moderator, reviewing posts based on:
    - Engagement metrics (likes + shares)
    - User flags
    - News content
    - Previous fact-checks
    """
    
    def __init__(
        self,
        checker_id: str,
        temperature: float = 0.3
    ):
        self.checker_id = checker_id
        self.temperature = temperature
        
        # Connect to database
        self.conn = sqlite3.connect(
            'database/simulation.db',
            timeout=30.0,
            isolation_level=None  # Enable autocommit mode
        )
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.cursor = self.conn.cursor()

        # Ensure required tables exist
        self._create_tables()

    def _create_tables(self):
        """Create necessary database tables if they don't exist."""
        # Create fact_checks table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS fact_checks (
                post_id TEXT NOT NULL,
                checker_id TEXT NOT NULL,
                verdict TEXT NOT NULL,
                explanation TEXT NOT NULL,
                confidence REAL NOT NULL,
                sources TEXT NOT NULL,
                groundtruth TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (post_id),
                FOREIGN KEY (post_id) REFERENCES posts(post_id)
            )
        ''')
        
        # Add columns to posts table - need to handle these separately
        try:
            self.cursor.execute('ALTER TABLE posts ADD COLUMN fact_check_status TEXT')
        except sqlite3.OperationalError:
            pass  # Column might already exist
        
        try:
            self.cursor.execute('ALTER TABLE posts ADD COLUMN fact_checked_at TIMESTAMP')
        except sqlite3.OperationalError:
            pass  # Column might already exist

    def get_posts_to_check(self, limit: int = 10, experiment_type: str = None) -> List[Post]:
        """
        Get posts that need fact-checking, prioritizing:
        1. Posts with high engagement (likes + shares)
        2. News posts
        3. Posts without existing fact-checks
        
        limit: number of posts to return
        experiment_type: type of experiment being run
        
        For hybrid_fact_checking experiment:
        1. Prioritize posts with community notes
        2. Then follow standard prioritization
        """
        if experiment_type == "hybrid_fact_checking":
            # First, get posts with community notes
            self.cursor.execute('''
                SELECT p.post_id, p.content, p.author_id, p.created_at,
                       p.num_likes, p.num_shares, p.num_flags, p.original_post_id,
                       p.num_comments, p.is_news, p.news_type, p.status,
                       p.takedown_timestamp, p.takedown_reason
                FROM posts p
                JOIN community_notes cn ON p.post_id = cn.post_id
                LEFT JOIN fact_checks fc ON p.post_id = fc.post_id
                WHERE fc.post_id IS NULL
                AND (p.status IS NULL OR p.status != 'taken_down')
                GROUP BY p.post_id
                ORDER BY 
                    COUNT(cn.note_id) DESC,
                    p.is_news DESC,
                    (p.num_likes + p.num_shares) DESC,
                    p.created_at DESC
                LIMIT ?
            ''', (limit,))
            
            posts_with_notes = [Post(*row) for row in self.cursor.fetchall()]
            
            # If we have enough posts with notes, return them
            if len(posts_with_notes) >= limit:
                # Load community notes for each post
                for post in posts_with_notes:
                    self.cursor.execute('''
                        SELECT note_id, content, author_id, helpful_ratings, not_helpful_ratings
                        FROM community_notes
                        WHERE post_id = ?
                    ''', (post.post_id,))
                    post.community_notes = [CommunityNote(*row) for row in self.cursor.fetchall()]
                return posts_with_notes
                
            # If we don't have enough posts with notes, get additional posts
            remaining_limit = limit - len(posts_with_notes)
            
            # Get posts without community notes
            self.cursor.execute('''
                SELECT p.post_id, p.content, p.author_id, p.created_at,
                       p.num_likes, p.num_shares, p.num_flags, p.original_post_id,
                       p.num_comments, p.is_news, p.news_type, p.status,
                       p.takedown_timestamp, p.takedown_reason
                FROM posts p
                LEFT JOIN community_notes cn ON p.post_id = cn.post_id
                LEFT JOIN fact_checks fc ON p.post_id = fc.post_id
                WHERE fc.post_id IS NULL
                AND cn.post_id IS NULL
                AND (p.status IS NULL OR p.status != 'taken_down')
                ORDER BY 
                    p.is_news DESC,
                    (p.num_likes + p.num_shares) DESC,
                    p.created_at DESC
                LIMIT ?
            ''', (remaining_limit,))
            
            additional_posts = [Post(*row) for row in self.cursor.fetchall()]
            return posts_with_notes + additional_posts
        
        # Default behavior for other experiment types
        self.cursor.execute('''
            SELECT p.post_id, p.content, p.author_id, p.created_at,
                   p.num_likes, p.num_shares, p.num_flags, p.original_post_id,
                   p.num_comments, p.is_news, p.news_type, p.status,
                   p.takedown_timestamp, p.takedown_reason
            FROM posts p
            LEFT JOIN fact_checks fc ON p.post_id = fc.post_id
            WHERE fc.post_id IS NULL
            AND (p.status IS NULL OR p.status != 'taken_down')
            ORDER BY 
                p.is_news DESC,
                (p.num_likes + p.num_shares) DESC,
                p.created_at DESC
            LIMIT ?
        ''', (limit,))

        posts = [Post(*row) for row in self.cursor.fetchall()]
        return posts

    def check_post(
        self,
        openai_client: OpenAI,
        engine: str,
        post: Post,
        experiment_type: str = None
    ) -> FactCheckVerdict:
        """
        Fact-check a post using the LLM.
        
        Args:
            openai_client: OpenAI client instance
            engine: Model engine to use
            post: Post object to fact-check
            experiment_type: Type of experiment being run
            
        Returns:
            FactCheckVerdict containing the fact-check results
        """
        prompt = self._create_fact_check_prompt(post)
        
        system_prompt = FactCheckerPrompts.get_system_prompt()
        verdict = Utils.generate_llm_response(
            openai_client=openai_client,
            engine=engine,
            prompt=prompt,
            system_message=system_prompt,
            temperature=self.temperature,
            response_model=FactCheckVerdict
        )
        
        self._record_verdict(post.post_id, verdict, experiment_type)
        return verdict

    def _create_fact_check_prompt(self, post: Post) -> str:
        """Create a prompt for fact-checking a specific post."""
        notes_text = ""
        for note in post.community_notes:
            notes_text += f"Note ID: {note.note_id}\nContent: {note.content}\nAuthor ID: {note.author_id}\n"

        prompt = FactCheckerPrompts.create_fact_check_prompt(
            post_content=post.content,
            community_notes=notes_text,
            engagement_metrics={
                "likes": post.num_likes,
                "shares": post.num_shares,
                "comments": post.num_comments
            },
        )
        
        return prompt

    def _record_verdict(self, post_id: str, verdict: FactCheckVerdict, experiment_type: str = None) -> None:
        """Record a fact-check verdict in the database and compare with groundtruth."""
        try:
            # Get the original post's groundtruth
            self.cursor.execute('''
                SELECT news_type, 
                       (SELECT COUNT(*) FROM community_notes WHERE post_id = p.post_id) as note_count
                FROM posts p
                WHERE p.post_id = ?
            ''', (post_id,))
            result = self.cursor.fetchone()
            groundtruth = result[0]
            note_count = result[1]

            # Insert the fact-check record with groundtruth
            self.cursor.execute('''
                INSERT INTO fact_checks (
                    post_id,
                    checker_id,
                    verdict,
                    explanation,
                    confidence,
                    sources,
                    groundtruth
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                post_id,
                self.checker_id,
                verdict.verdict,
                verdict.explanation,
                verdict.confidence,
                json.dumps(verdict.sources),
                groundtruth
            ))

            # Update the post's fact-check status
            self.cursor.execute('''
                UPDATE posts 
                SET fact_check_status = ?,
                    fact_checked_at = CURRENT_TIMESTAMP
                WHERE post_id = ?
            ''', (verdict.verdict, post_id))

            # If confidence is high and verdict is false, mark post as taken down
            # For hybrid fact checking, take down posts more aggressively when there are community notes
            should_take_down = False
            
            if verdict.verdict == "false":
                if experiment_type == "hybrid_fact_checking" and note_count > 0:
                    # In hybrid mode with community notes, take down with lower confidence threshold
                    should_take_down = verdict.confidence >= 0.7
                else:
                    # Standard threshold for other cases
                    should_take_down = verdict.confidence >= 0.9
            
            if should_take_down:
                self.cursor.execute('''
                    UPDATE posts 
                    SET status = 'taken_down'
                    WHERE post_id = ?
                ''', (post_id,))
                
                takedown_reason = "Fact-checked as false with high confidence"
                if experiment_type == "hybrid_fact_checking" and note_count > 0:
                    takedown_reason += " and supported by community notes"
                
                self.cursor.execute('''
                    UPDATE posts 
                    SET takedown_reason = ?
                    WHERE post_id = ?
                ''', (takedown_reason, post_id))

                logging.info(f"Fact checker {self.checker_id} marked post {post_id} as {verdict.verdict} and took it down")
            else:
                logging.info(f"Fact checker {self.checker_id} marked post {post_id} as {verdict.verdict}")

        except sqlite3.IntegrityError:
            logging.warning(f"Post {post_id} has already been fact-checked")
            self.conn.rollback()
        except Exception as e:
            logging.error(f"Error recording fact-check verdict: {e}")
            self.conn.rollback()
            raise

    def __del__(self):
        """Cleanup database connection."""
        if hasattr(self, 'conn'):
            self.conn.close() 

def main():
    """
    Main function to run the fact checker.
    """
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Create fact checker instance
    checker = FactChecker(
        checker_id="main_checker",
        temperature=0.3
    )
    
    posts = checker.get_posts_to_check(limit=5)
    
    if not posts:
        print("No posts found that need fact-checking.")
        return
    
    print(f"Found {len(posts)} posts to fact-check.")
    
    # Process each post
    for i, post in enumerate(posts, 1):
        print(f"\nChecking post {i}/{len(posts)} (ID: {post.post_id})")
        print(f"Content: {post.content[:100]}...")  # Show first 100 chars
        
        try:
            verdict = checker.check_post(
                openai_client=openai_client,
                engine="gpt-4o",
                post=post
            )
        
            print("\nVerdict:")
            print(f"  Status: {verdict.verdict}")
            print(f"  Confidence: {verdict.confidence:.2%}")
            print(f"  Explanation: {verdict.explanation}")
            print(f"  Sources: {', '.join(verdict.sources)}")
                
        except Exception as e:
            logging.error(f"Error checking post {post.post_id}: {e}")
            continue    

if __name__ == "__main__":
    main() 