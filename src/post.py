from datetime import datetime
from comment import Comment
from typing import List

class CommunityNote:
    def __init__(self, note_id, content, author_id, helpful_ratings, not_helpful_ratings):
        self.note_id = note_id
        self.content = content
        self.author_id = author_id
        self.helpful_ratings = helpful_ratings
        self.not_helpful_ratings = not_helpful_ratings
        
    @property
    def is_visible(self) -> bool:
        """Note becomes visible when it has sufficient helpful ratings"""
        # return self.helpful_ratings >= 3 and self.helpful_ratings > self.not_helpful_ratings * 2
        return True
class Post:
    """
    A DTO representing a post in the simulation.
    """
    def __init__(self, 
        post_id: str, 
        content: str, 
        author_id: str, 
        created_at: datetime,
        num_likes: int = 0,
        num_shares: int = 0,
        num_flags: int = 0,
        num_comments: int = 0,
        original_post_id: str = None,
        is_news: bool = False,
        news_type: str = None,
        status: str = 'active',
        takedown_timestamp: datetime = None,
        takedown_reason: str = None,
        comments: list[Comment] = None,
    ):
        self.post_id = post_id
        self.content = content
        self.author_id = author_id
        self.created_at = created_at
        self.num_likes = num_likes
        self.num_shares = num_shares
        self.num_flags = num_flags
        self.num_comments = num_comments
        self.original_post_id = original_post_id
        self.is_news = is_news
        self.news_type = news_type
        self.status = status
        self.takedown_timestamp = takedown_timestamp
        self.takedown_reason = takedown_reason
        self.comments = comments or []
        self.community_notes: List[CommunityNote] = []
    
    @property
    def is_flagged(self) -> bool:
        """Returns True if the post has been flagged multiple times."""
        return self.num_flags >= 2

    def to_dict(self) -> dict:
        """Convert post to dictionary for serialization."""
        return {
            'post_id': self.post_id,
            'content': self.content,
            'author_id': self.author_id,
            'created_at': self.created_at.isoformat(),
            'num_likes': self.num_likes,
            'num_shares': self.num_shares,
            'num_flags': self.num_flags,
            'num_comments': self.num_comments,
            'original_post_id': self.original_post_id,
            'is_news': self.is_news,
            'news_type': self.news_type,
            'status': self.status,
            'takedown_timestamp': self.takedown_timestamp.isoformat() if self.takedown_timestamp else None,
            'takedown_reason': self.takedown_reason,
            'comments': [comment.to_dict() for comment in self.comments]
        }
    
    