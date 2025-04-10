import json
import sqlite3
import logging
from openai import OpenAI
from comment import Comment
from post import Post, CommunityNote
from utils import Utils
import time
from agent_memory import AgentMemory
from prompts import AgentPrompts
from pydantic import BaseModel
from typing import List, Literal, Optional


class FeedAction(BaseModel):
    """
    An action to take on a post in the feed.
    This class is used to validate the action taken by the agent, and to enforce structured output from the LLM.
    """
    action: Literal[
        "like-post", "share-post", "flag-post", "follow-user", 
        "unfollow-user", "comment-post", "like-comment", "ignore",
        "add-note", "rate-note"  # New actions
    ]
    target: Optional[str] = None  # Made optional since 'ignore' doesn't need a target
    content: Optional[str] = None
    # reasoning: Optional[str] = None # only include reasoning if include_reasoning is true
    note_rating: Optional[Literal["helpful", "not-helpful"]] = None  # For rating notes


class FeedReaction(BaseModel):
    """
    A reaction to the feed.
    This class is used to enforce structured output from the LLM.
    """
    actions: List[FeedAction]


class AgentUser:
    """
    An LLM agent in the simulation.
    
    Attributes (a dict): 
        user_id: unique id of the agent.
        persona: The persona / backstory of the agent.
            - background: The background of the agent.
            - labels: The interest labels of the agent.
    
    Actions:
        create_post: Create a post.
        like_post: Like a post.
        share_post: Share a post (retweet)
        flag_post: Flag a post (report)
        follow_user: Follow another user.
        unfollow_user: Unfollow another user.   
    """
    MEMORY_TYPE_INTERACTION = 'interaction'
    MEMORY_TYPE_REFLECTION = 'reflection'
    VALID_MEMORY_TYPES = {MEMORY_TYPE_INTERACTION, MEMORY_TYPE_REFLECTION}
    
    def __init__(
        self, 
        user_id: str, 
        user_config: dict,
        temperature: float = 0.8,
        is_news_agent: bool = False,
        experiment_config: dict = None  # Add experiment_config parameter
    ):
        self.user_id = user_id
        self.persona = user_config['persona']
        self.is_news_agent = is_news_agent
        # connect to the database
        self.conn = sqlite3.connect(
            'database/simulation.db', 
            timeout=30.0,  # 30 second timeout
            isolation_level=None  # Enable autocommit mode
        )
        self.conn.execute('PRAGMA journal_mode=WAL')  # Use Write-Ahead Logging
        self.cursor = self.conn.cursor()
        self.temperature = temperature

        # Initialize memory manager with the same connection
        self.memory = AgentMemory(
            user_id=self.user_id,
            conn=self.conn,
            persona=self.persona,
            memory_decay_rate=0.1
        )
        
        # Store the config for later use
        self.user_config = user_config
        # Store experiment config and type
        self.experiment_config = experiment_config or {}
        self.experiment_type = self.experiment_config.get('experiment', {}).get('type', 'default')

    def create_post(self, content: str, is_news: bool = False, news_type: str = None, status: str = 'active') -> str:
        """
        Create a post for the user.
        """
        post_id = Utils.generate_formatted_id("post")
        
        # Create the post, marking it as news if from news agent
        self.cursor.execute('''
            INSERT INTO posts (post_id, content, author_id, is_news, news_type, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (post_id, content, self.user_id, is_news, news_type, status))
        
        # Record the post creation action
        self.cursor.execute('''
            INSERT INTO user_actions (user_id, action_type, target_id, content)
            VALUES (?, 'post', ?, ?)
        ''', (self.user_id, post_id, content))
        
        self.conn.commit()
        logging.info(f"{'News Agent' if self.is_news_agent else 'User'} {self.user_id} created a post: {post_id} | {content}")
        
        return post_id

    def like_post(self, post_id: str) -> None:
        """
        Like a post.
        """
        # if an user has already liked this post, don't like it again
        self.cursor.execute('''
            SELECT COUNT(*) FROM user_actions 
            WHERE user_id = ? AND action_type = 'like' AND target_id = ?
        ''', (self.user_id, post_id))
        if self.cursor.fetchone()[0] > 0:
            logging.info(f"User {self.user_id} already liked post {post_id}")
            return
        
        # Update post likes count
        self.cursor.execute('UPDATE posts SET num_likes = num_likes + 1 WHERE post_id = ?', (post_id,))
        
        # Update author's total likes received
        self.cursor.execute('''
            UPDATE users 
            SET total_likes_received = total_likes_received + 1
            WHERE user_id = (
                SELECT author_id 
                FROM posts 
                WHERE post_id = ?
            )
        ''', (post_id,))
        
        self.conn.commit()
        logging.info(f"User {self.user_id} liked post {post_id}")

    def create_comment(self, post_id: str, content: str) -> str:
        """
        Create a comment on a post and update the post's comment count.
        """
        comment_id = Utils.generate_formatted_id("comment")
        
        # Get post author
        self.cursor.execute('SELECT author_id FROM posts WHERE post_id = ?', (post_id,))
        post_author = self.cursor.fetchone()[0]
        
        # Insert the comment
        self.cursor.execute('''
            INSERT INTO comments (comment_id, content, post_id, author_id)
            VALUES (?, ?, ?, ?)
        ''', (comment_id, content, post_id, self.user_id))
        
        # Update post's comment count
        self.cursor.execute('''
            UPDATE posts 
            SET num_comments = num_comments + 1 
            WHERE post_id = ?
        ''', (post_id,))
        
        # Update total comments received for post author
        self.cursor.execute('''
            UPDATE users 
            SET total_comments_received = total_comments_received + 1
            WHERE user_id = ?
        ''', (post_author,))
        
        self.conn.commit()
        logging.info(f"User {self.user_id} commented on post {post_id}: {content}")
        return comment_id

    def like_comment(self, comment_id: str) -> None:
        """
        Like a comment and track the action.
        """
        try:
            # Check if user already liked this comment
            self.cursor.execute('''
                SELECT COUNT(*) FROM user_actions 
                WHERE user_id = ? AND action_type = 'like_comment' 
                AND target_id = ?
            ''', (self.user_id, comment_id))
            if self.cursor.fetchone()[0] > 0:
                logging.info(f"User {self.user_id} already liked comment {comment_id}")
                return
            
            # Update database
            self.cursor.execute('''
                UPDATE comments 
                SET num_likes = num_likes + 1 
                WHERE comment_id = ?
            ''', (comment_id,))
            
            self.conn.commit()
            logging.info(f"User {self.user_id} liked comment {comment_id}")
            
        except sqlite3.OperationalError as e:
            self.conn.rollback()
            logging.error(f"Database operation error: {e}")
        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            logging.error(f"Integrity error: {e}")

    def share_post(self, post_id: str) -> None:
        """
        Share a post (repost it to user's own feed).
        """
        # Check if user already shared this post
        self.cursor.execute('''
            SELECT COUNT(*) FROM user_actions 
            WHERE user_id = ? AND action_type = 'share' AND target_id = ?
        ''', (self.user_id, post_id))
        if self.cursor.fetchone()[0] > 0:
            logging.info(f"User {self.user_id} already shared post {post_id}")
            return
        
        # Get the original post content
        self.cursor.execute('SELECT content, author_id FROM posts WHERE post_id = ?', (post_id,))
        original_post = self.cursor.fetchone()
        if not original_post:
            logging.warning(f"Post {post_id} not found")
            return
        
        original_content, original_author = original_post
        
        # Create a new post as a share
        new_post_id = Utils.generate_formatted_id("post")
        # shared_content = f"Reposted from @{original_author}: {original_content}"
        shared_content = original_content # hide the @original_author
        # Insert the new post
        self.cursor.execute('''
            INSERT INTO posts (post_id, content, author_id, original_post_id)
            VALUES (?, ?, ?, ?)
        ''', (new_post_id, shared_content, self.user_id, post_id))
        
        # Increment share count on original post
        self.cursor.execute('UPDATE posts SET num_shares = num_shares + 1 WHERE post_id = ?', (post_id,))
        
        # Update total shares received for original author
        self.cursor.execute('''
            UPDATE users 
            SET total_shares_received = total_shares_received + 1
            WHERE user_id = ?
        ''', (original_author,))
        
        self.conn.commit()
        logging.info(f"User {self.user_id} shared post {post_id}")
    
    def flag_post(self, post_id: str) -> None:
        """
        Flag a post.
        """
        # Check if user already flagged this post
        self.cursor.execute('''
            SELECT COUNT(*) FROM user_actions 
            WHERE user_id = ? AND action_type = 'flag' AND target_id = ?
        ''', (self.user_id, post_id))
        if self.cursor.fetchone()[0] > 0:
            logging.info(f"User {self.user_id} already flagged post {post_id}")
            return
        
        self.cursor.execute('UPDATE posts SET num_flags = num_flags + 1 WHERE post_id = ?', (post_id,))
        self.conn.commit()
        logging.info(f"User {self.user_id} flagged post {post_id}")
    
    def follow_user(self, target_user_id: str) -> None:
        """
        Follow another user.
        """
        # First check if already following
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 1 FROM follows 
            WHERE follower_id = ? AND followed_id = ?
        ''', (self.user_id, target_user_id))
        
        if cursor.fetchone() is not None:
            logging.info(f"User {self.user_id} already follows {target_user_id}")
            return
        
        # If not already following, create the relationship
        cursor.execute('''
            INSERT INTO follows (follower_id, followed_id)
            VALUES (?, ?)
        ''', (self.user_id, target_user_id))
        
        # Update follower count for target user
        cursor.execute('''
            UPDATE users 
            SET follower_count = follower_count + 1 
            WHERE user_id = ?
        ''', (target_user_id,))
        
        self.conn.commit()
        logging.info(f"User {self.user_id} followed {target_user_id}")

    def unfollow_user(self, target_user_id: str) -> None:
        """
        Unfollow a user.
        """
        # First check if actually following
        self.cursor.execute('''
            SELECT 1 FROM follows 
            WHERE follower_id = ? AND followed_id = ?
        ''', (self.user_id, target_user_id))
        
        if self.cursor.fetchone() is None:
            logging.info(f"User {self.user_id} was not following {target_user_id}")
            return
            
        # delete the follow from the database
        self.cursor.execute('''
            DELETE FROM follows
            WHERE follower_id = ? AND followed_id = ?
        ''', (self.user_id, target_user_id))
        
        # Update follower count for target user
        self.cursor.execute('''
            UPDATE users 
            SET follower_count = follower_count - 1 
            WHERE user_id = ?
        ''', (target_user_id,))
        
        self.conn.commit()
        logging.info(f"User {self.user_id} unfollowed user {target_user_id}")
    
    
    def ignore(self) -> None:
        """
        Record that the agent chose to ignore their feed.
        """
        logging.info(f"User {self.user_id} ignored their feed")
        
        
    def _generate_post_content(
        self, 
        openai_client: OpenAI, 
        engine: str, 
        max_tokens: int = 512
    ) -> str:
        """
        Generate a post content for the user.
        Args:
            openai_client: The OpenAI client to use for generating content.
            engine: The engine to use for generation.
            max_tokens: The maximum number of tokens to generate.
        Returns:
            content: The content of the post.
        """
        time.sleep(0.5)  # Rate limiting
        prompt = self._create_post_prompt()
        
        system_prompt = AgentPrompts.get_system_prompt()
        return Utils.generate_llm_response(
            openai_client=openai_client,
            engine=engine,
            prompt=prompt,
            system_message=system_prompt,
            temperature=self.temperature,
            max_tokens=max_tokens
        )


    def _get_recent_posts(self, limit=5) -> list[str]:
        """
        Get the user's recent posts.
        Returns:
            list[str]: List of recent post contents
        """
        self.cursor.execute('''
            SELECT content FROM posts 
            WHERE author_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (self.user_id, limit))
        return [row[0] for row in self.cursor.fetchall()]


    def _create_post_prompt(self) -> str:
        """
        Create a post prompt for the user.
        Returns:
            prompt: The prompt for the post.
        """
        # Get recent posts from the user
        recent_posts = self._get_recent_posts()
        recent_posts_text = "\n".join([f"- {post}" for post in recent_posts]) if recent_posts else ""
        
        # Get recent feed posts
        feed = self.get_feed(experiment_config=self.experiment_config, time_step=None)
        feed_text = "\n".join([
            f"- {post.content}"
            f"{' [BREAKING NEWS]' if hasattr(post, 'is_news') and post.is_news else ''}"
            f"{' [FLAGGED BY COMMUNITY]' if hasattr(post, 'is_flagged') and post.is_flagged else ''}"
            + (f"\n  Community Notes:\n" + "\n".join([
                f"  • note_id: {note.note_id} | {note.content} (Helpful: {note.helpful_ratings}, Not Helpful: {note.not_helpful_ratings})"
                for note in post.community_notes if note.is_visible
            ]) if any(note.is_visible for note in post.community_notes) else "")
            for post in feed
        ]) if feed else ""
        
        # Get relevant memories
        relevant_memories = self.memory.get_relevant_memories("interaction", limit=3)
        memories_text = "\n".join([f"- {mem['content']}" for mem in relevant_memories])
        
        prompt = AgentPrompts.create_post_prompt(
            self.persona,
            memories_text,
            recent_posts_text,
            feed_text
        )
        token_count = Utils.estimate_token_count(prompt)
        logging.info(f"Post generation prompt token count: {token_count}")

        return prompt


    def get_feed(self, experiment_config: dict, time_step=None):
        """
        Get the user's feed and track post exposures.
        """
        num_non_followed_posts = experiment_config['feed']['num_non_followed_posts']
        num_followed_posts = experiment_config['feed']['num_followed_posts']
        
        # Get posts from followed users (including news posts)
        self.cursor.execute('''
            SELECT p.post_id, p.content, p.author_id, p.created_at, 
                   p.num_likes, p.num_shares, p.num_flags, p.original_post_id 
            FROM posts p
            JOIN follows f ON p.author_id = f.followed_id
            WHERE f.follower_id = ? 
            AND (p.status IS NULL OR p.status != 'taken_down')
            ORDER BY p.created_at DESC
            LIMIT ?
        ''', (self.user_id, num_followed_posts))
        followed_posts = {Post(*row) for row in self.cursor.fetchall()}

        # Get news and non-followed posts together
        self.cursor.execute('''
            SELECT p.post_id, p.content, p.author_id, p.created_at, 
                   p.num_likes, p.num_shares, p.num_flags, p.original_post_id 
            FROM posts p
            WHERE (p.author_id NOT IN (
                SELECT followed_id FROM follows
                WHERE follower_id = ?
            )
            OR p.is_news = TRUE)
            AND p.author_id != ?
            AND (p.status IS NULL OR p.status != 'taken_down')
            ORDER BY p.created_at DESC
            LIMIT ?
        ''', (self.user_id, self.user_id, num_non_followed_posts))
        other_posts = {Post(*row) for row in self.cursor.fetchall()}

        # Combine all posts into a set to remove any remaining duplicates
        final_feed = sorted(followed_posts | other_posts, key=lambda x: x.created_at, reverse=True)
        
        # Fetch comments for each post
        for post in final_feed:
            self.cursor.execute('''
                SELECT c.comment_id, c.content, c.post_id, c.author_id, 
                       c.created_at, c.num_likes
                FROM comments c
                WHERE c.post_id = ?
                ORDER BY c.num_likes DESC, c.created_at DESC
                LIMIT 3
            ''', (post.post_id,))
            
            post.comments = [Comment(*row) for row in self.cursor.fetchall()]

        # Fetch community notes for each post
        for post in final_feed:
            self.cursor.execute('''
                SELECT note_id, content, author_id, helpful_ratings, not_helpful_ratings
                FROM community_notes
                WHERE post_id = ?
                ORDER BY helpful_ratings DESC
            ''', (post.post_id,))
            
            post.community_notes = [CommunityNote(*row) for row in self.cursor.fetchall()]

        # Add fact-check information to posts
        for post in final_feed:
            self.cursor.execute('''
                SELECT verdict, explanation, confidence
                FROM fact_checks
                WHERE post_id = ?
            ''', (post.post_id,))
            fact_check = self.cursor.fetchone()
            if fact_check:
                post.fact_check_verdict = fact_check[0]
                post.fact_check_explanation = fact_check[1]
                post.fact_check_confidence = fact_check[2]

        # Track exposures for all posts in the final feed
        if time_step is not None:
            for post in final_feed:
                self.cursor.execute('''
                    INSERT OR IGNORE INTO feed_exposures (user_id, post_id, time_step)
                    VALUES (?, ?, ?)
                ''', (self.user_id, post.post_id, time_step))
            self.conn.commit()

        return final_feed

    def get_news_only_feed(self, experiment_config: dict, time_step=None):
        """Optimized feed generation for news-only simulation."""
        total_news_posts = experiment_config['feed']['total_news_posts']
        
        # Get all news posts, prioritizing followed sources
        self.cursor.execute('''
            SELECT p.post_id, p.content, p.author_id, p.created_at, 
                   p.num_likes, p.num_shares, p.num_flags, p.original_post_id,
                   CASE WHEN f.followed_id IS NOT NULL THEN 1 ELSE 0 END AS is_followed
            FROM posts p
            LEFT JOIN follows f ON p.author_id = f.followed_id AND f.follower_id = ?
            WHERE p.is_news = TRUE
            AND p.author_id != ?
            AND (p.status IS NULL OR p.status != 'taken_down')
            ORDER BY is_followed DESC, p.created_at DESC
            LIMIT ?
        ''', (self.user_id, self.user_id, total_news_posts))
        
        news_posts = [Post(*row[:-1]) for row in self.cursor.fetchall()]  # Exclude is_followed from Post constructor
        
        # Combine all posts into a set to remove any remaining duplicates
        final_feed = sorted(news_posts, key=lambda x: x.created_at, reverse=True)
        
        # Fetch comments for each post
        for post in final_feed:
            self.cursor.execute('''
                SELECT c.comment_id, c.content, c.post_id, c.author_id, 
                       c.created_at, c.num_likes
                FROM comments c
                WHERE c.post_id = ?
                ORDER BY c.num_likes DESC, c.created_at DESC
                LIMIT 3
            ''', (post.post_id,))
            
            post.comments = [Comment(*row) for row in self.cursor.fetchall()]

        # Fetch community notes for each post
        for post in final_feed:
            self.cursor.execute('''
                SELECT note_id, content, author_id, helpful_ratings, not_helpful_ratings
                FROM community_notes
                WHERE post_id = ?
                ORDER BY helpful_ratings DESC
            ''', (post.post_id,))
            
            post.community_notes = [CommunityNote(*row) for row in self.cursor.fetchall()]

        # Add fact-check information to posts
        for post in final_feed:
            self.cursor.execute('''
                SELECT verdict, explanation, confidence
                FROM fact_checks
                WHERE post_id = ?
            ''', (post.post_id,))
            fact_check = self.cursor.fetchone()
            if fact_check:
                post.fact_check_verdict = fact_check[0]
                post.fact_check_explanation = fact_check[1]
                post.fact_check_confidence = fact_check[2]

        # Track exposures for all posts in the final feed
        if time_step is not None:
            for post in final_feed:
                self.cursor.execute('''
                    INSERT OR IGNORE INTO feed_exposures (user_id, post_id, time_step)
                    VALUES (?, ?, ?)
                ''', (self.user_id, post.post_id, time_step))
            self.conn.commit()

        return final_feed

    def react_to_feed(
        self, 
        openai_client: OpenAI, 
        engine: str, 
        feed: list[Post]
    ):
        """
        React to the feed using LLM-driven decisions.
        News agents don't react to feeds - they only publish news.
        
        Args:
            openai_client: The OpenAI client to use for generating content.
            engine: The engine to use for generation.
            feed: A list of Post objects representing the user's feed.
        """
        # Skip feed reactions for news agents
        if self.is_news_agent:
            return
        
        # Get the reasoning configuration
        include_reasoning = self.experiment_config.get('experiment', {}).get('settings', {}).get('include_reasoning', False)
        
        prompt = self._create_feed_reaction_prompt(feed)
        
        logging.info(f"Feed reaction prompt:\n {prompt}")
        
        system_prompt = AgentPrompts.get_system_prompt()
        
        # Create a dynamic FeedAction class based on whether reasoning is included
        if include_reasoning:
            class FeedActionWithReasoning(BaseModel):
                action: Literal[
                    "like-post", "share-post", "flag-post", "follow-user", 
                    "unfollow-user", "comment-post", "like-comment", "ignore",
                    "add-note", "rate-note"
                ]
                target: Optional[str] = None
                content: Optional[str] = None
                reasoning: Optional[str] = None
                note_rating: Optional[Literal["helpful", "not-helpful"]] = None
            
            class FeedReactionWithReasoning(BaseModel):
                actions: List[FeedActionWithReasoning]
            
            response_model = FeedReactionWithReasoning
        else:
            response_model = FeedReaction
        
        reaction = Utils.generate_llm_response(
            openai_client=openai_client,
            engine=engine,
            prompt=prompt,
            system_message=system_prompt,
            temperature=self.temperature,
            response_model=response_model,
        )
        
        self._process_reaction(reaction, feed)
        
        # Check for reflection after processing reactions
        self.cursor.execute('''
            SELECT COUNT(*) FROM agent_memories 
            WHERE user_id = ? AND memory_type = 'interaction'
        ''', (self.user_id,))
        
        if self.cursor.fetchone()[0] % 2 == 0:  # every 2 interactions
            self.memory.reflect(openai_client, engine, self.temperature)

    def _create_feed_reaction_prompt(self, feed: list[Post]) -> str:
        """
        Create a prompt for the LLM to decide how to react to the feed.
        """
        feed_content = "\n".join([
            f"post_id: {post.post_id} | content: {post.content} "
            f"(by User {post.author_id})"
            # f"{' [NEWS]' if hasattr(post, 'is_news') and post.is_news else ''}"
            # f"{' [FLAGGED BY COMMUNITY]' if hasattr(post, 'is_flagged') and post.is_flagged else ''}"
            + (f"\nFACT CHECK: {post.fact_check_verdict.upper()} "
               f"(Confidence: {post.fact_check_confidence:.0%})\n"
               f"Explanation: {post.fact_check_explanation}" 
               if hasattr(post, 'fact_check_verdict') else "")
            + f"\nComments:\n" + "\n".join([
                f"- comment_id: {comment.comment_id} | content: {comment.content} (by User {comment.author_id})"
                for comment in post.comments[:3]
            ])
            + (f"\n  Community Notes:\n" + "\n".join([
                f"  • note_id: {note.note_id} | content: {note.content} (Helpful: {note.helpful_ratings}, Not Helpful: {note.not_helpful_ratings})"
                for note in post.community_notes[:3] if note.is_visible
            ]) if any(note.is_visible for note in post.community_notes[:3]) else "")
            for i, post in enumerate(feed)
        ])
        
        relevant_memories = self.memory.get_relevant_memories("interaction", limit=5)
        memories_text = "\n".join([f"- {mem['content']}" for mem in relevant_memories])
        
        reflection_memories = self.memory.get_relevant_memories("reflection", limit=1)
        reflections_text = "\n".join([f"- {mem['content']}" for mem in reflection_memories])

        # Get the reasoning configuration
        include_reasoning = self.experiment_config.get('experiment', {}).get('settings', {}).get('include_reasoning', False)

        prompt = AgentPrompts.create_feed_reaction_prompt(
            self.persona,
            memories_text,
            feed_content,
            reflections_text,
            self.experiment_type,
            include_reasoning
        )
        
        # logging.info(f"Feed reaction prompt: {prompt}")

        return prompt

    def _process_reaction(self, reaction: FeedReaction, feed: list[Post]):
        """Process the reaction generated by the LLM with enhanced validation and memory creation."""
        try:
            # Pre-compute valid targets with better note validation
            valid_targets = {
                'post': {post.post_id for post in feed},
                'user': {post.author_id for post in feed},
                'comment': {comment.comment_id for post in feed for comment in post.comments},
                'note': {
                    note.note_id 
                    for post in feed 
                    for note in post.community_notes 
                    if note.is_visible
                }
            }
            
            # Define action validation rules
            action_rules = {
                'like-post': {'target_type': 'post', 'needs_content': False},
                'share-post': {'target_type': 'post', 'needs_content': False},
                'flag-post': {'target_type': 'post', 'needs_content': False},
                'follow-user': {'target_type': 'user', 'needs_content': False},
                'unfollow-user': {'target_type': 'user', 'needs_content': False},
                'comment-post': {'target_type': 'post', 'needs_content': True},
                'like-comment': {'target_type': 'comment', 'needs_content': False},
                'ignore': {'target_type': None, 'needs_content': False},
                'add-note': {'target_type': 'post', 'needs_content': True},
                'rate-note': {'target_type': 'note', 'needs_content': False}
            }
            
            # Add validation for note ratings
            def validate_note_rating(target: str, note_rating: Optional[str]) -> tuple:
                """
                Validate that a note rating action is valid.
                Returns:
                    tuple: (is_valid: bool, reason: str)
                """
                if target not in valid_targets['note']:
                    return False, "Invalid note target"
                if note_rating not in ['helpful', 'not-helpful']:
                    return False, "Invalid rating type"
                # Check if user has already rated this note
                self.cursor.execute('''
                    SELECT COUNT(*) FROM note_ratings 
                    WHERE user_id = ? AND note_id = ?
                ''', (self.user_id, target))
                if self.cursor.fetchone()[0] > 0:
                    return False, "User has already rated this note"
                return True, "Valid rating"

            processed_actions = set()
            
            for action_data in reaction.actions:
                action = action_data.action
                target = action_data.target
                content = action_data.content
                # Check if reasoning is included in the action data
                action_reasoning = getattr(action_data, 'reasoning', None)
                
                # Validate target exists in feed
                rules = action_rules[action]
                if rules['target_type'] and target not in valid_targets[rules['target_type']]:
                    logging.warning(f"Invalid target: {target} for action: {action}")
                    continue
                    
                # Prevent duplicates and self-actions
                action_key = f"{action}:{target}"
                if action_key in processed_actions or (
                    action in {'follow-user', 'unfollow-user'} and target == self.user_id
                ):
                    logging.warning(f"Duplicate or self-action: {action_key}")
                    continue
                
                processed_actions.add(action_key)
                
                try:
                    # Execute action
                    time.sleep(0.5)  # Rate limiting
                    
                    # Check if reasoning should be included in database
                    include_reasoning = self.experiment_config.get('experiment', {}).get('settings', {}).get('include_reasoning', False)
                    
                    if action == 'comment-post' or action == 'add-note':
                        if include_reasoning and action_reasoning:
                            self.cursor.execute('''
                                INSERT INTO user_actions (user_id, action_type, target_id, content, reasoning)
                                VALUES (?, ?, ?, ?, ?)
                            ''', (self.user_id, action.replace('-', '_'), target, content, action_reasoning))
                        else:
                            self.cursor.execute('''
                                INSERT INTO user_actions (user_id, action_type, target_id, content)
                                VALUES (?, ?, ?, ?)
                            ''', (self.user_id, action.replace('-', '_'), target, content))
                    elif action == 'ignore':
                        if include_reasoning and action_reasoning:
                            self.cursor.execute('''
                                INSERT INTO user_actions (user_id, action_type, reasoning)
                                VALUES (?, 'ignore', ?)
                            ''', (self.user_id, action_reasoning))
                        else:
                            self.cursor.execute('''
                                INSERT INTO user_actions (user_id, action_type)
                                VALUES (?, 'ignore')
                            ''', (self.user_id,))
                        logging.info(f"User {self.user_id} ignored their feed")
                    else:
                        if include_reasoning and action_reasoning:
                            self.cursor.execute('''
                                INSERT INTO user_actions (user_id, action_type, target_id, reasoning)
                                VALUES (?, ?, ?, ?)
                            ''', (self.user_id, action.replace('-', '_'), target, action_reasoning))
                        else:
                            self.cursor.execute('''
                                INSERT INTO user_actions (user_id, action_type, target_id)
                                VALUES (?, ?, ?)
                            ''', (self.user_id, action.replace('-', '_'), target))
                    
                    # Execute the actual action
                    if action == 'comment-post':
                        self.create_comment(target, content)
                    elif action == 'add-note':
                        self.add_community_note(target, content)
                    elif action == 'rate-note':
                        is_valid, reason = validate_note_rating(target, action_data.note_rating)
                        if not is_valid:
                            logging.warning(f"Invalid note rating: {target} with rating {action_data.note_rating} - {reason}")
                            continue
                        is_helpful = action_data.note_rating == "helpful"
                        self.rate_community_note(target, is_helpful)
                    elif action == 'ignore':
                        self.ignore()
                    else:
                        getattr(self, action.replace('-', '_'))(target)  # Call the action method, e.g. self.like_post(target)
                    
                    # Create memory for this specific action
                    memory_content = f"I {action.replace('-', ' ')} {target}"
                    if content:
                        memory_content += f": '{content}'"
                    if include_reasoning and action_reasoning:
                        memory_content += f" because {action_reasoning}"
                    
                    # Calculate importance for this specific action
                    importance = 0.5  # Base importance
                    if action in ['comment-post', 'share-post']:
                        importance = 0.7  # Higher importance for more engaging actions
                    if 'news' in str(feed).lower():
                        importance += 0.2  # Higher importance for news interactions
                        
                    self.memory.add_memory(memory_content, 'interaction', importance)
                    
                except Exception as e:
                    logging.error(f"Error executing action {action}: {e}")

        except Exception as e:
            logging.error(f"Error processing reaction: {e}")
            raise

    def add_community_note(self, post_id: str, content: str) -> str:
        """
        Add a community note to a post.
        Returns the note_id.
        """
        note_id = Utils.generate_formatted_id("note")
        
        self.cursor.execute('''
            INSERT INTO community_notes (
                note_id, post_id, author_id, content, 
                helpful_ratings, not_helpful_ratings
            )
            VALUES (?, ?, ?, ?, 0, 0)
        ''', (note_id, post_id, self.user_id, content))
        
        # Record the note creation action
        self.cursor.execute('''
            INSERT INTO user_actions (user_id, action_type, target_id, content)
            VALUES (?, 'add_note', ?, ?)
        ''', (self.user_id, post_id, content))
        
        self.conn.commit()
        logging.info(f"User {self.user_id} added note to post {post_id}: {content}")
        return note_id

    def rate_community_note(self, note_id: str, is_helpful: bool) -> None:
        """
        Rate a community note as helpful or not helpful.
        """
        # First check if the note exists
        self.cursor.execute('''
            SELECT COUNT(*) FROM community_notes 
            WHERE note_id = ?
        ''', (note_id,))
        
        if self.cursor.fetchone()[0] == 0:
            logging.warning(f"Note {note_id} does not exist")
            return
        
        # Check if user already rated this note
        self.cursor.execute('''
            SELECT COUNT(*) FROM note_ratings 
            WHERE user_id = ? AND note_id = ?
        ''', (self.user_id, note_id))
        
        if self.cursor.fetchone()[0] > 0:
            logging.info(f"User {self.user_id} already rated note {note_id}")
            return
        
        # Add rating
        rating_type = "helpful" if is_helpful else "not_helpful"
        self.cursor.execute('''
            INSERT INTO note_ratings (note_id, user_id, rating)
            VALUES (?, ?, ?)
        ''', (note_id, self.user_id, rating_type))
        
        # Update note rating counts
        field = "helpful_ratings" if is_helpful else "not_helpful_ratings"
        self.cursor.execute(f'''
            UPDATE community_notes 
            SET {field} = {field} + 1 
            WHERE note_id = ?
        ''', (note_id,))
        
        self.conn.commit()
        logging.info(f"User {self.user_id} rated note {note_id} as {rating_type}")
