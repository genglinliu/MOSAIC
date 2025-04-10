from datetime import datetime
import uuid
import logging
import networkx as nx
import matplotlib.pyplot as plt
import sqlite3
import os
import pandas as pd
from typing import Optional, Type, Union
from openai import OpenAI
from pydantic import BaseModel
import matplotlib
from tenacity import retry, stop_after_attempt, wait_exponential
from deprecated import deprecated
import ollama

class Utils:
    @staticmethod
    def configure_logging(engine: str): 
        """Configure logging for the simulation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('experiment_outputs/logs', exist_ok=True)
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=f'experiment_outputs/logs/{timestamp}-{engine}.log', 
                            filemode='w')
        
        # Add a stream handler to also print to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Silence web request related loggers
        for logger_name in ["httpx", "requests", "urllib3"]:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
            
    @staticmethod
    def generate_formatted_id(prefix: str, conn: Optional[sqlite3.Connection] = None) -> str:
        """Generate a formatted ID with the given prefix and last 6 digits of a UUID.
        Handles UUID collisions if a database connection is provided."""
        
        # Map of prefixes to their corresponding tables
        prefix_table_map = {
            "user": "users",
            "post": "posts",
            "comment": "comments",
            "note": "community_notes",
            "memory": "agent_memories",
        }
        
        if conn is None:
            # If no connection provided, just return a new ID (original behavior)
            full_uuid = uuid.uuid4()
            last_6_digits = str(full_uuid)[-6:]
            return f"{prefix}-{last_6_digits}"
        
        # Get the corresponding table name for the prefix
        table_name = prefix_table_map.get(prefix)
        if not table_name:
            logging.warning(f"Unknown prefix '{prefix}', collision detection disabled")
            full_uuid = uuid.uuid4()
            last_6_digits = str(full_uuid)[-6:]
            return f"{prefix}-{last_6_digits}"
        
        # Keep trying until we find a unique ID
        while True:
            full_uuid = uuid.uuid4()
            last_6_digits = str(full_uuid)[-6:]
            new_id = f"{prefix}-{last_6_digits}"
            
            # Check if ID exists in the corresponding table
            cursor = conn.cursor()
            id_column = f"{prefix}_id"
            cursor.execute(f"SELECT 1 FROM {table_name} WHERE {id_column} = ?", (new_id,))
            
            if cursor.fetchone() is None:
                # ID is unique, we can use it
                return new_id
            
            logging.warning(f"UUID collision detected for {prefix}, generating new ID...")

    @staticmethod
    @deprecated(reason="We no longer need to visualize the network in this way.")
    def visualize_network(conn, action: str, timestamp: str):
        """Visualize the network graph of users and their follow relationships."""
        # Set up matplotlib
        matplotlib.use('Agg', force=True)
        plt.clf()
        plt.close('all')
        
        # Get users and create graph
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users")
        users = [row[0] for row in cursor.fetchall()]
        
        if not users:
            logging.warning("No users found in database - skipping visualization")
            return
        
        # Create and populate graph
        G = nx.DiGraph()
        G.add_nodes_from(users)
        
        cursor.execute("""
            SELECT user_id, target_id 
            FROM user_actions 
            WHERE action_type = 'follow'
        """)
        edges = [(follower, followed) 
                for follower, followed in cursor.fetchall() 
                if follower in users and followed in users]
        G.add_edges_from(edges)
        
        # Create and configure plot
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw network elements
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
        nx.draw_networkx_edges(G, pos, 
                              edge_color='gray',
                              arrows=True, 
                              arrowsize=10,
                              width=2,
                              min_target_margin=15,
                              connectionstyle='arc3,rad=0.2')
        
        # Add labels
        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        
        # Configure plot
        ax.set_title("User Network Graph", pad=20)
        ax.margins(0.2)
        
        # Add border box
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.set_frame_on(True)
        
        # Save as PNG only
        output_dir = f'experiment_outputs/plots/{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/network_graph-{action}.png'
        
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close('all')


    @staticmethod
    @retry(
        stop=stop_after_attempt(1000),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
        before_sleep=lambda retry_state: logging.warning(
            f"Retry attempt {retry_state.attempt_number} after error: {retry_state.outcome.exception()}"
        )
    )
    def generate_llm_response(
        openai_client: OpenAI, 
        engine: str, 
        prompt: str, 
        system_message: str, 
        temperature: float,
        response_model: Optional[Type[BaseModel]] = None,
        max_tokens: int = 4096,
        # stop: list[str] = ['\n']
    ) -> Union[str, BaseModel]:
        """Generate a response from the LLM."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Use parse for structured output if model is provided
        if response_model:
            if "gpt" in engine: # OpenAI api
                completion = openai_client.beta.chat.completions.parse(
                    model=engine,
                    messages=messages,
                    response_format=response_model,
                    temperature=temperature,
                    timeout=15,  # 15 seconds timeout
                    frequency_penalty=1.6,
                    presence_penalty=1.6
                )
                return completion.choices[0].message.parsed
            else: # ollama
                response = ollama.chat(
                    model=engine,
                    messages=messages,
                    format="json",
                    options={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "schema": response_model.model_json_schema()
                    }
                )
                # Parse the JSON response into the response model
                return response_model.model_validate_json(response.message.content)
        
        # Regular completion without response format - for Post generation
        else:
            if "gpt" in engine: # OpenAI api
                completion = openai_client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # # stop=stop,
                    # timeout=20,  # 20 seconds timeout
                    # frequency_penalty=1.6,
                    # presence_penalty=1.6
                )
                return completion.choices[0].message.content
            else: # ollama
                response = ollama.chat(
                    model=engine,
                    messages=messages,
                    options={
                        "temperature": temperature,
                    }
                )
                return response.message.content

    @staticmethod
    def update_user_influence(conn: sqlite3.Connection):
        """Update influence scores for all users."""
        # Get user metrics into a DataFrame
        df = pd.read_sql_query('''
            SELECT 
                user_id,
                follower_count,
                total_likes_received,
                total_shares_received,
                total_comments_received
            FROM users
        ''', conn)
        
        # Calculate normalized scores (handling division by zero)
        metrics = {
            'follower_count': 0.4,
            'total_likes_received': 0.3,
            'total_shares_received': 0.2,
            'total_comments_received': 0.1
        }
        
        influence_scores = pd.Series(0.0, index=df.index)
        
        for metric, weight in metrics.items():
            max_val = df[metric].max()
            if max_val > 0:  # Only normalize if we have non-zero values
                influence_scores += (df[metric] / max_val) * weight
        
        # Update the database with new scores
        df['influence_score'] = influence_scores
        df['is_influencer'] = influence_scores > 0.5
        
        # Update the database
        for _, row in df.iterrows():
            conn.execute('''
                UPDATE users 
                SET influence_score = ?,
                    is_influencer = ?,
                    last_influence_update = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (round(float(row['influence_score']), 3), bool(row['is_influencer']), row['user_id']))
        
        conn.commit()
        
        # Log influencer status changes
        influencers = df[df['is_influencer']][['user_id', 'influence_score']].sort_values('influence_score', ascending=False)
        
        if not influencers.empty:
            logging.info("\nCurrent Influencers:")
            for _, row in influencers.iterrows():
                logging.info(f"User {row['user_id']}: Influence Score = {row['influence_score']:.3f}")

    @staticmethod
    def evaluate_fact_checker_performance(conn: sqlite3.Connection):
        """
        Evaluate the performance of the fact checker on identifying fake news.
        Calculates accuracy, precision, recall, and F1 score.
        
        Args:
            conn: SQLite database connection
        """
        cursor = conn.cursor()
        
        # Get confusion matrix values
        cursor.execute('''
            SELECT 
                p.news_type,
                fc.verdict,
                COUNT(*) as count
            FROM posts p
            JOIN fact_checks fc ON p.post_id = fc.post_id
            GROUP BY p.news_type, fc.verdict
        ''')
        
        # Initialize confusion matrix values
        true_positives = 0  # Correctly identified fake news
        false_positives = 0  # Real news incorrectly marked as fake
        false_negatives = 0  # Fake news incorrectly marked as real
        true_negatives = 0  # Correctly identified real news
        
        for news_type, verdict, count in cursor.fetchall():
            is_fake_verdict = verdict in ('false', 'misleading', 'unverified')
            
            if news_type == 'fake' and is_fake_verdict:
                true_positives += count
            elif news_type == 'fake' and not is_fake_verdict:
                false_negatives += count
            elif news_type == 'real' and is_fake_verdict:
                false_positives += count
            elif news_type == 'real' and not is_fake_verdict:
                true_negatives += count
        
        total_samples = true_positives + false_positives + false_negatives + true_negatives
        
        if total_samples > 0:
            # Calculate metrics
            accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print("\nFact Checker Performance Metrics:")
            print(f"Total posts fact-checked: {total_samples}")
            print("\nConfusion Matrix:")
            print(f"True Positives (Correct fake news detection): {true_positives}")
            print(f"False Positives (Real news marked as fake): {false_positives}")
            print(f"True Negatives (Correct real news detection): {true_negatives}")
            print(f"False Negatives (Missed fake news): {false_negatives}")
            print("\nMetrics:")
            print(f"Accuracy: {accuracy:.1%}")
            print(f"Precision: {precision:.1%}")
            print(f"Recall: {recall:.1%}")
            print(f"F1 Score: {f1:.1%}")
        else:
            print("\nNo posts have been fact-checked yet.")

    @staticmethod
    def print_simulation_stats(conn: sqlite3.Connection):
        """Print the simulation statistics."""
        cursor = conn.cursor()
        
        # Get total number of users
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        # Get total number of posts
        cursor.execute("SELECT COUNT(*) FROM posts")
        total_posts = cursor.fetchone()[0]
        # Get total number of actions
        cursor.execute("SELECT COUNT(*) FROM user_actions")
        total_actions = cursor.fetchone()[0]
        # Get breakdown of action types
        cursor.execute("SELECT action_type, COUNT(*) FROM user_actions GROUP BY action_type")
        action_breakdown = cursor.fetchall()
        
        logging.info("Simulation Statistics:")
        logging.info(f"Total users: {total_users}")
        logging.info(f"Total posts: {total_posts}")
        logging.info(f"Total user actions: {total_actions}")
        logging.info("Action breakdown:")
        for action_type, count in action_breakdown:
            logging.info(f"  {action_type}: {count}")

        # Add community notes statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_notes,
                SUM(CASE WHEN helpful_ratings >= 3 AND helpful_ratings > not_helpful_ratings * 2 
                    THEN 1 ELSE 0 END) as visible_notes,
                AVG(helpful_ratings) as avg_helpful_ratings,
                AVG(not_helpful_ratings) as avg_not_helpful_ratings
            FROM community_notes
        ''')
        note_stats = cursor.fetchone()
        
        print("\nCommunity Notes Statistics:")
        print(f"Total Number of Notes Created: {note_stats[0]}")
        print(f"Number of Visible Notes: {note_stats[1]}")
        print(f"Average Helpful Ratings: {note_stats[2]:.2f}" if note_stats[2] is not None else "N/A")
        print(f"Average Not Helpful Ratings: {note_stats[3]:.2f}" if note_stats[3] is not None else "N/A")

        # Add fact checker evaluation
        Utils.evaluate_fact_checker_performance(conn)

    @staticmethod
    def get_influence_stats(conn: sqlite3.Connection, user_id: str) -> dict:
        """
        Get the current influence statistics for a user.
        
        Args:
            conn: SQLite database connection
            user_id: The ID of the user to get stats for
            
        Returns:
            dict: Dictionary containing influence statistics, or None if user not found
        """
        cursor = conn.cursor()
        cursor.execute('''
            SELECT follower_count, total_likes_received, 
                   total_shares_received, total_comments_received,
                   influence_score, is_influencer
            FROM users
            WHERE user_id = ?
        ''', (user_id,))
        
        stats = cursor.fetchone()
        if stats:
            return {
                'followers': stats[0],
                'total_likes': stats[1],
                'total_shares': stats[2],
                'total_comments': stats[3],
                'influence_score': stats[4],
                'is_influencer': stats[5]
            }
        return None

    @staticmethod
    def estimate_token_count(prompt: str) -> int:
        """
        Estimate the number of tokens in a string. This is a rough approximation.
        GPT models generally treat words, punctuation, and spaces as tokens.
        
        Args:
            prompt: The input string to estimate tokens for
            
        Returns:
            int: Estimated number of tokens
        """
        # Split into words
        words = prompt.split()
        
        # Count punctuation marks that are likely to be separate tokens
        punctuation_count = sum(1 for char in prompt if char in '.,!?;:()[]{}""\'')
        
        # Basic estimate: each word is roughly one token
        # Add punctuation count and some overhead for spaces and special characters
        estimated_tokens = len(words) + punctuation_count
        
        # Add 20% overhead for potential subword tokenization
        estimated_tokens = int(estimated_tokens * 1.2)
        
        return max(1, estimated_tokens)  # Ensure at least 1 token is returned
