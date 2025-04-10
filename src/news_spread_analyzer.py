from typing import Dict, List, Tuple
import pandas as pd
import networkx as nx
import logging

class NewsSpreadAnalyzer:
    """
    Analyzes the spread of news posts throughout the social network.
    Tracks views, interaction depth (hops), and breadth of engagement.
    """
    def __init__(self, db_manager, config):
        self.db_manager = db_manager
        self.conn = db_manager.get_connection()
        self.flag_threshold = config.get('moderation', {}).get('flag_threshold', 1)
        self.note_threshold = config.get('moderation', {}).get('note_threshold', 1)
        self.content_moderation = config.get('moderation', {}).get('content_moderation', True)

    def track_news_views(self, news_post_id: str, time_step: int) -> int:
        """
        Counts total number of users who have been exposed to the news post
        up to and including this time step.
        """
        query = """
            SELECT COUNT(DISTINCT user_id)
            FROM feed_exposures
            WHERE post_id = ? AND time_step <= ?
        """
        cursor = self.conn.execute(query, (news_post_id, time_step))
        return cursor.fetchone()[0]

    def calculate_diffusion_depth(self, news_post_id: int) -> int:
        """
        Calculates the maximum number of sharing hops from the original news post.
        Returns the maximum depth of sharing chain.
        """
        query = """
            WITH RECURSIVE sharing_chain AS (
                -- Base case: original news post
                SELECT post_id as id, original_post_id as parent_post_id, 0 as depth
                FROM posts
                WHERE post_id = ?
                
                UNION ALL
                
                -- Recursive case: shared posts
                SELECT p.post_id, p.original_post_id, sc.depth + 1
                FROM posts p
                JOIN sharing_chain sc ON p.original_post_id = sc.id
            )
            SELECT MAX(depth) FROM sharing_chain
        """
        cursor = self.conn.execute(query, (news_post_id,))
        return cursor.fetchone()[0] or 0

    def calculate_diffusion_breadth(self, news_post_id: int) -> Dict[str, int]:
        """
        Calculates the breadth of diffusion by counting users who interacted
        with the news post through various actions (including community notes).
        """
        metrics = {}
        
        # Count likes, shares, and flags using user_actions table
        query = """
            SELECT action_type, COUNT(DISTINCT user_id)
            FROM user_actions
            WHERE target_id = ?
            AND action_type IN ('like_post', 'share_post', 'flag_post')
            GROUP BY action_type
        """
        cursor = self.conn.execute(query, (news_post_id,))
        
        # Map action types to metric names
        action_mapping = {
            'like_post': 'num_likes',
            'share_post': 'num_shares',
            'flag_post': 'num_flags'
        }
        
        for action_type, count in cursor.fetchall():
            metrics[action_mapping[action_type]] = count
        
        # Initialize missing metrics with 0
        for metric in ['num_likes', 'num_shares', 'num_flags']:
            if metric not in metrics:
                metrics[metric] = 0
        
        # Count comments
        query = """
            SELECT COUNT(DISTINCT author_id)
            FROM comments
            WHERE post_id = ?
        """
        cursor = self.conn.execute(query, (news_post_id,))
        metrics['num_comments'] = cursor.fetchone()[0]
        
        # Count unique community note authors
        query = """
            SELECT COUNT(DISTINCT author_id)
            FROM community_notes
            WHERE post_id = ?
        """
        cursor = self.conn.execute(query, (news_post_id,))
        metrics['num_notes'] = cursor.fetchone()[0]
        
        # Count unique note raters
        query = """
            SELECT COUNT(DISTINCT nr.user_id)
            FROM note_ratings nr
            JOIN community_notes cn ON nr.note_id = cn.note_id
            WHERE cn.post_id = ?
        """
        cursor = self.conn.execute(query, (news_post_id,))
        metrics['num_note_ratings'] = cursor.fetchone()[0]
        
        # Add total interactions
        metrics['total_interactions'] = sum(metrics.values())
        
        return metrics

    def store_spread_metrics(self, news_post_id: int, metrics: Dict) -> None:
        """
        Stores the spread metrics in the database for a specific time step.
        Preserves historical data by using post_id + time_step as composite key.
        """
        query = """
            INSERT INTO spread_metrics (
                post_id, time_step, views, diffusion_depth,
                num_likes, num_shares, num_flags,
                num_comments, num_notes, num_note_ratings,
                total_interactions, should_takedown, takedown_reason,
                takedown_executed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.conn.execute(query, (
            news_post_id,
            metrics['time_step'],
            metrics['views'],
            metrics['diffusion_depth'],
            metrics['num_likes'],
            metrics['num_shares'],
            metrics['num_flags'],
            metrics['num_comments'],
            metrics['num_notes'],
            metrics['num_note_ratings'],
            metrics['total_interactions'],
            metrics['should_takedown'],
            metrics['takedown_reason'],
            metrics['takedown_executed']
        ))
        self.conn.commit()

    def should_take_down_post(
        self, 
        news_post_id: int,
        ) -> Tuple[bool, str]:
        """
        Determines if a news post should be taken down based on negative feedback.
        
        Args:
            news_post_id: The ID of the news post to check
        """
        if not self.content_moderation:
            return False, "Content moderation is disabled"
        
        # Check number of flags
        query = """
            SELECT COUNT(DISTINCT user_id)
            FROM user_actions
            WHERE target_id = ? AND action_type = 'flag_post'
        """
        cursor = self.conn.execute(query, (news_post_id,))
        flag_count = cursor.fetchone()[0]
        
        if flag_count >= self.flag_threshold:
            return True, f"Received {flag_count} flags, exceeding threshold of {self.flag_threshold}"
            
        # Check number of critical community notes
        query = """
            SELECT COUNT(DISTINCT note_id)
            FROM community_notes
            WHERE post_id = ?
        """
        cursor = self.conn.execute(query, (news_post_id,))
        critical_notes = cursor.fetchone()[0]
        
        if critical_notes >= self.note_threshold:
            return True, f"Received {critical_notes} critical notes, exceeding threshold of {self.note_threshold}"
            
        return False, "Post does not meet takedown criteria"

    def take_down_post(self, news_post_id: int, reason: str) -> bool:
        """
        Executes the takedown of a news post from the platform.
        
        Args:
            news_post_id: The ID of the news post to take down
            reason: The reason for taking down the post
            
        Returns:
            bool: True if takedown was successful, False otherwise
        """
        try:
            # Update post status to taken down
            update_query = """
                UPDATE posts 
                SET status = 'taken_down',
                    takedown_timestamp = CURRENT_TIMESTAMP,
                    takedown_reason = ?
                WHERE post_id = ?
            """
            self.conn.execute(update_query, (reason, news_post_id))
            
            # Log the takedown action
            log_query = """
                INSERT INTO moderation_logs (
                    post_id, action_type, reason, timestamp
                ) VALUES (?, 'takedown', ?, CURRENT_TIMESTAMP)
            """
            self.conn.execute(log_query, (news_post_id, reason))
            
            self.conn.commit()
            logging.info(f"Post {news_post_id} has been taken down. Reason: {reason}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Failed to take down post {news_post_id}: {str(e)}")
            return False

    def analyze_spread(self, news_post_id: int, time_step: int) -> Dict:
        """
        Analyzes and returns all spread metrics for a given news post at a specific time step.
        Also stores the metrics in the database and checks if post should be taken down.
        """
        views = self.track_news_views(news_post_id, time_step)
        depth = self.calculate_diffusion_depth(news_post_id)
        breadth_metrics = self.calculate_diffusion_breadth(news_post_id)
        
        # Check if post should be taken down
        should_takedown, reason = self.should_take_down_post(news_post_id)
        
        # Execute takedown if necessary
        takedown_executed = False
        if should_takedown:
            takedown_executed = self.take_down_post(news_post_id, reason)
        
        metrics = {
            'time_step': time_step,
            'views': views,
            'diffusion_depth': depth,
            'should_takedown': should_takedown,
            'takedown_reason': reason,
            'takedown_executed': takedown_executed,
            **breadth_metrics
        }
        
        self.store_spread_metrics(news_post_id, metrics)
        return metrics 