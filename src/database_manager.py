import os
import logging
import sqlite3
import time
import shutil
import json
from typing import Dict, List

class DatabaseManager:
    def __init__(self, db_path: str, reset_db: bool = True):
        self.db_path = db_path
        self.reset_db = reset_db
        
        # Close any existing connections first
        try:
            sqlite3.connect(self.db_path).close()
        except:
            pass
        
        # Wait a moment to ensure the connection is fully closed
        time.sleep(1)
        
        # Initialize database connection
        try:
            self.conn = sqlite3.connect(self.db_path, timeout=30.0)
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.OperationalError as e:
            logging.error(f"Could not establish database connection: {e}")
            raise
        
        # Initialize database
        if self.reset_db:
            self.reset_database()
        else:
            self.create_tables()

    def reset_database(self):
        """Reset the database and create new tables."""
        logging.info("Resetting database...")
        
        try:
            # Close the existing connection
            self.conn.close()
            
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                logging.info("Existing database removed.")
            else:
                logging.info("No existing database found.")
            
            # Recreate the connection
            self.conn = sqlite3.connect(self.db_path, timeout=30.0)
            self.conn.execute("PRAGMA foreign_keys = ON")
            
            self.create_tables()
            logging.info("New tables created.")
            
        except Exception as e:
            logging.error(f"Error resetting database: {str(e)}")
            raise

    def create_tables(self):
        """Create the database tables."""
        cursor = self.conn.cursor()
        
        if self.reset_db:   
            # Drop tables in reverse order of dependencies
            cursor.execute("DROP TABLE IF EXISTS spread_metrics")
            cursor.execute("DROP TABLE IF EXISTS comments")
            cursor.execute("DROP TABLE IF EXISTS user_objectives")
            cursor.execute("DROP TABLE IF EXISTS user_actions")
            cursor.execute("DROP TABLE IF EXISTS follows")
            cursor.execute("DROP TABLE IF EXISTS posts")
            cursor.execute("DROP TABLE IF EXISTS users")
            cursor.execute("DROP TABLE IF EXISTS fact_checks")
        
        # Create tables
        tables = {
            'users': '''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    persona TEXT,
                    background_labels JSON,
                    creation_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    follower_count INTEGER DEFAULT 0,
                    total_likes_received INTEGER DEFAULT 0,
                    total_shares_received INTEGER DEFAULT 0,
                    total_comments_received INTEGER DEFAULT 0,
                    influence_score FLOAT DEFAULT 0.0,
                    is_influencer BOOLEAN DEFAULT FALSE,
                    last_influence_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'posts': '''
                CREATE TABLE IF NOT EXISTS posts (
                    post_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    num_likes INTEGER DEFAULT 0,
                    num_shares INTEGER DEFAULT 0,
                    num_flags INTEGER DEFAULT 0,
                    num_comments INTEGER DEFAULT 0,
                    original_post_id TEXT,
                    is_news BOOLEAN DEFAULT FALSE,
                    news_type TEXT,
                    status TEXT CHECK(status IN ('active', 'taken_down')),
                    takedown_timestamp TIMESTAMP,
                    takedown_reason TEXT,
                    fact_check_status TEXT,
                    fact_checked_at TIMESTAMP,
                    FOREIGN KEY (author_id) REFERENCES users(user_id),
                    FOREIGN KEY (original_post_id) REFERENCES posts(post_id)
                )
            ''',
            'moderation_logs': '''
            CREATE TABLE IF NOT EXISTS moderation_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id INTEGER,
                    action_type TEXT,
                    reason TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (post_id) REFERENCES posts(post_id)
                )
            ''',
            'community_notes': '''
                CREATE TABLE IF NOT EXISTS community_notes (
                    note_id TEXT PRIMARY KEY,
                    post_id TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    helpful_ratings INTEGER DEFAULT 0,
                    not_helpful_ratings INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (post_id) REFERENCES posts(post_id),
                    FOREIGN KEY (author_id) REFERENCES users(user_id)
                )
            ''',
            'note_ratings': '''
                CREATE TABLE IF NOT EXISTS note_ratings (
                    note_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating TEXT CHECK(rating IN ('helpful', 'not_helpful')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (note_id, user_id),
                    FOREIGN KEY (note_id) REFERENCES community_notes(note_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''',
            'user_actions': '''
                CREATE TABLE IF NOT EXISTS user_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    target_id TEXT,
                    content TEXT,
                    reasoning TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''',
            'follows': '''
                CREATE TABLE IF NOT EXISTS follows (
                    follower_id TEXT NOT NULL,
                    followed_id TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (follower_id, followed_id),
                    FOREIGN KEY (follower_id) REFERENCES users(user_id),
                    FOREIGN KEY (followed_id) REFERENCES users(user_id)
                )
            ''',
            'comments': '''
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    post_id TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    num_likes INTEGER DEFAULT 0,
                    FOREIGN KEY (post_id) REFERENCES posts(post_id),
                    FOREIGN KEY (author_id) REFERENCES users(user_id)
                )
            ''',
            'agent_memories': '''
                CREATE TABLE IF NOT EXISTS agent_memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance_score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    decay_factor FLOAT DEFAULT 1.0,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''',
            'spread_metrics': '''
                CREATE TABLE IF NOT EXISTS spread_metrics (
                    post_id TEXT NOT NULL,
                    time_step INTEGER NOT NULL,
                    views INTEGER NOT NULL,
                    diffusion_depth INTEGER NOT NULL,
                    num_likes INTEGER NOT NULL,
                    num_shares INTEGER NOT NULL,
                    num_flags INTEGER NOT NULL,
                    num_comments INTEGER NOT NULL,
                    num_notes INTEGER NOT NULL,
                    num_note_ratings INTEGER NOT NULL,
                    total_interactions INTEGER NOT NULL,
                    should_takedown BOOLEAN,
                    takedown_reason TEXT,
                    takedown_executed BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (post_id, time_step),
                    FOREIGN KEY (post_id) REFERENCES posts(post_id)
                )
            ''',
            'feed_exposures': '''
                CREATE TABLE IF NOT EXISTS feed_exposures (
                    user_id TEXT NOT NULL,
                    post_id TEXT NOT NULL,
                    time_step INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, post_id, time_step),
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (post_id) REFERENCES posts(post_id)
                )
            ''',
            'fact_checks': '''
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
            '''
        }
        
        for table_name, create_statement in tables.items():
            cursor.execute(create_statement)
        
        self.conn.commit()
        logging.info("Database tables created successfully.")

    def save_simulation_db(self, timestamp: str):
        """Save a timestamped copy of the simulation database."""
        # Close the connection to ensure all data is written
        self.conn.close()
        
        # Create archive directory if it doesn't exist
        archive_dir = f"experiment_outputs/database_copies"
        os.makedirs(archive_dir, exist_ok=True)
        
        # Copy the database file
        archived_db = f"{archive_dir}/{timestamp}.db"
        shutil.copy2(self.db_path, archived_db)
        logging.info(f"Saved simulation database to {archived_db}")
        
    
    def add_user(self, user_id: str, user_config: dict):
        """Add a new user to the database.
        
        Args:
            user_id: Unique identifier for the user
            user_config: Standardized user configuration containing:
                - background_labels: Dict of arbitrary user attributes
                - persona: Dict with 'background' and 'labels' keys
        """
        # Convert background labels to JSON string
        background_labels = user_config.get('background_labels', {})
        
        self.conn.execute('''
            INSERT INTO users (
                user_id,
                persona,
                background_labels
            )
            VALUES (?, ?, json(?))
        ''', (
            user_id,
            str(user_config['persona']),
            json.dumps(background_labels)
        ))
            
        self.conn.commit()
        logging.info(f"Added user {user_id} to database with {len(background_labels)} background labels.")

    def get_connection(self):
        """Get the database connection."""
        return self.conn

def get_schema_info(db_path: str) -> Dict[str, List[tuple]]:
    """
    Get schema information for all tables in the database.
    Returns a dictionary with table names as keys and list of column information as values.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = {}
    for (table_name,) in tables:
        # Get column information for each table
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        schema_info[table_name] = columns
    
    conn.close()
    return schema_info

def print_schema(schema_info: Dict[str, List[tuple]]):
    """
    Print a text representation of the schema.
    """
    for table_name, columns in schema_info.items():
        print(f"\n=== {table_name} ===")
        for col in columns:
            col_id, col_name, col_type, notnull, default, pk = col
            pk_str = "PRIMARY KEY" if pk else ""
            null_str = "NOT NULL" if notnull else "NULL"
            default_str = f"DEFAULT {default}" if default else ""
            print(f"  {col_name}: {col_type} {pk_str} {null_str} {default_str}".strip())

if __name__ == "__main__":
    db_path = "/Users/genglinliu/Documents/GitHub/social-simulation/database/simulation.db"  
    
    # Get schema information
    schema_info = get_schema_info(db_path)
    
    # Print text representation
    print_schema(schema_info)
    