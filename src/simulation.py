from datetime import datetime
import logging
import os
import random
from openai import OpenAI
from keys import OPENAI_API_KEY
from utils import Utils
import json
from homophily_analysis import HomophilyAnalysis
from tqdm import tqdm
from news_manager import NewsManager
from database_manager import DatabaseManager
from user_manager import UserManager
from news_spread_analyzer import NewsSpreadAnalyzer
from fact_checker import FactChecker


class Simulation:
    """
    A simulation of a social media platform.
    """
    def __init__(self, config: dict):
        self.config = config  # Store the entire config dictionary
        self.reset_db = config.get('reset_db', True)
        self.num_users = config['num_users']
        self.engine = config['engine']
        self.generate_own_post = config.get('generate_own_post', True)  # New parameter with default True
        
        # Generate timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize database manager
        self.db_manager = DatabaseManager('database/simulation.db', self.reset_db)
        self.conn = self.db_manager.get_connection()
        self.db_path = self.db_manager.db_path
        
        # Replace user management with UserManager
        self.user_manager = UserManager(config, self.db_manager)
        self.users = self.user_manager.users
        
        # Create news agent
        self.news_manager = NewsManager(self.config, self.conn)
        self.news_agent = self.news_manager.news_agent
        
        # Create initial follows
        self.user_manager.create_initial_follows()
        
        # Initialize the OpenAI client
        if self.engine.startswith("gpt"):
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.openai_client = OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama'
            )
        
        # Initialize news spread analyzer with config
        self.news_spread_analyzer = NewsSpreadAnalyzer(self.db_manager, self.config)
        
        # Initialize fact checker for both third-party and hybrid approaches
        self.experiment_type = config.get('experiment', {}).get('type', 'none')
        self.experiment_settings = config.get('experiment', {}).get('settings', {})
        if self.experiment_type in ["third_party_fact_checking", "hybrid_fact_checking"]:
            self.fact_checker = FactChecker(
                checker_id="main_checker",
                temperature=self.experiment_settings.get('fact_checker_temperature', 0.3)
            )
        else:
            self.fact_checker = None
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        

    def run(self, num_time_steps: int):
        """Run the simulation."""
        new_user_config = self.config.get('new_users', {})
        add_new_users_probability = new_user_config.get('add_probability', 0.9)
        new_user_follow_probability = new_user_config.get('follow_probability', 0.0)
        news_start_step = self.config.get('news_injection', {}).get('start_step', 1)
        
        # Save a copy of the experiment configuration
        # Create directory if it doesn't exist
        config_dir = f"experiment_outputs/configs"
        os.makedirs(config_dir, exist_ok=True)
        
        # Save the configuration with the same timestamp used for other outputs
        config_path = f"{config_dir}/{self.timestamp}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        logging.info(f"Saved experiment configuration to {config_path}")
        
        # Get fact checking settings if applicable for both third-party and hybrid
        fact_check_limit = self.experiment_settings.get('posts_per_step', 5) if self.experiment_type in ["third_party_fact_checking", "hybrid_fact_checking"] else 0
        
        # Track all injected news post IDs
        injected_news_posts = []
        
        # Add tqdm progress bar
        progress_bar = tqdm(range(num_time_steps), desc="Running simulation")
        
        for step in progress_bar:
            logging.info(f"Time step: {step + 1}")
            
            # Update progress bar description with current step
            progress_bar.set_description(f"Time step {step + 1}/{num_time_steps}")
            
            # Inject news at specified step
            if step >= news_start_step:
                news_post_ids = self.news_manager.inject_news()
                if news_post_ids:
                    injected_news_posts.extend(news_post_ids)
                    logging.info(f"Injected news post (ID: {news_post_ids}) at step {step + 1}")
            
            # Update user addition to use UserManager
            if random.random() < add_new_users_probability:
                num_new_users = 2
                self.user_manager.add_random_users(num_new_users, follow_probability=new_user_follow_probability)
                # Update our reference to users
                self.users = self.user_manager.users
            
            # Each user creates a post (only if generate_own_post is True)
            if self.generate_own_post:
                for i, user in enumerate(self.users):
                    if i % 5 == 0:
                        logging.info(f"User #{i} creating post, at step {step + 1}")
                    content = user._generate_post_content(self.openai_client, self.engine, max_tokens=256)
                    user.create_post(content, is_news=False, news_type=None, status='active')
            
            # Each user reacts to their feed
            for i, user in enumerate(self.users):
                if i % 5 == 0:
                    logging.info(f"User #{i} reacting to feed, at step {step + 1}")
                if self.experiment_type in ["third_party_fact_checking", "hybrid_fact_checking"]:
                    feed = user.get_news_only_feed(experiment_config=self.config, time_step=step)
                else:
                    feed = user.get_feed(experiment_config=self.config, time_step=step)
                
                user.react_to_feed(self.openai_client, self.engine, feed)
            
            # Analyze and log spread for all injected news posts
            for news_post_id in injected_news_posts:
                spread_metrics = self.news_spread_analyzer.analyze_spread(news_post_id, step)
                logging.info(f"News ID {news_post_id} spread metrics - {spread_metrics}")
            
            # Update influence scores
            Utils.update_user_influence(self.conn)
            
            # Run experiment-specific checks at the end of each step
            if self.experiment_type in ["third_party_fact_checking", "hybrid_fact_checking"] and self.fact_checker:
                posts_to_check = self.fact_checker.get_posts_to_check(
                    limit=fact_check_limit,
                    experiment_type=self.experiment_type
                )
                for post in posts_to_check:
                    try:
                        verdict = self.fact_checker.check_post(
                            openai_client=self.openai_client,
                            engine=self.engine,
                            post=post,
                            experiment_type=self.experiment_type
                        )
                        logging.info(f"Fact checked post {post.post_id}: {verdict.verdict} ({verdict.confidence:.0%})")
                    except Exception as e:
                        logging.error(f"Error fact-checking post {post.post_id}: {e}")
                        continue
            
            logging.info("")  # Add a newline for readability between time steps
        
        # Print the simulation statistics
        logging.info("\nSimulation complete. Printing statistics...")
        Utils.print_simulation_stats(self.conn)
        
        # Then save and close the database as the last step
        self.db_manager.save_simulation_db(timestamp=self.timestamp)
        
        # Run homophily analysis after simulation completes
        homophily_analyzer = HomophilyAnalysis(self.db_path)
        homophily_analyzer.run_analysis(output_dir=f"experiment_outputs/homophily_analysis/{self.timestamp}")

    