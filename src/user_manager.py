from agent_user import AgentUser
from utils import Utils
import random
import jsonlines
import logging
import time
from database_manager import DatabaseManager
class UserManager:
    def __init__(self, config: dict, db_manager: DatabaseManager):
        self.experiment_config = config
        self.num_users = config['num_users']
        self.db_manager = db_manager
        self.conn = db_manager.get_connection()
        # Add tracking for used configs
        self.used_configs = set()
        self.all_user_configs = None
        self.users = self.create_users()
        
    def load_agent_configs(self) -> list[dict]:
        """Load agent configurations either from a JSONL file or generate them dynamically."""
        try:
            generation_method = self.experiment_config.get('agent_config_generation', 'file')
            
            if generation_method == 'file':
                # Load all configs if not already loaded
                if self.all_user_configs is None:
                    config_file = self.experiment_config.get('agent_config_path')
                    if not config_file:
                        raise ValueError("No agent_config_path specified in experiment config")
                    
                    with jsonlines.open(config_file) as reader:
                        self.all_user_configs = list(reader)
                
                # Reset if we've used all configs
                if len(self.used_configs) >= len(self.all_user_configs):
                    self.used_configs.clear()
                
                # Get available configs
                available_configs = [
                    config for i, config in enumerate(self.all_user_configs)
                    if i not in self.used_configs
                ]
                
                # Sample from available configs
                num_to_sample = min(self.num_users, len(available_configs))
                configs = random.sample(available_configs, num_to_sample)
                
                # Track newly used configs
                for config in configs:
                    self.used_configs.add(self.all_user_configs.index(config))
                
            # elif generation_method == 'agent_bank':
            #     from agent_config_generator_persona import generate_agent_configs_agent_bank
            #     configs = generate_agent_configs_agent_bank(num_agents=self.num_users)
                
            # elif generation_method == 'fine_persona':
            #     from agent_config_generator_persona import generate_agent_configs_fine_persona
            #     configs = generate_agent_configs_fine_persona(num_agents=self.num_users)
                
            # elif generation_method == 'simple':
            #     from agent_config_generator_persona import generate_agent_configs_simple
            #     configs = generate_agent_configs_simple(num_agents=self.num_users)
                
            else:
                raise ValueError(f"Unknown agent config generation method: {generation_method}")
            
            # Standardize all configs to ensure consistent format
            standardized_configs = []
            for config in configs:
                standardized_config = {
                    'background_labels': {},
                    'persona': {},
                }
                
                # Handle background information
                standardized_config['background_labels'] = {
                    **{k:v for k,v in config.items()
                    if k not in ['persona', 'id']}
                }
                
                # Handle persona information
                if 'persona' in config:
                    standardized_config['persona'] = config['persona']
                else:
                    logging.warning("No persona found in config")
                
                standardized_configs.append(standardized_config)
            
            logging.info(f"Generated {len(standardized_configs)} agent configurations using {generation_method} method")
            return standardized_configs
                
        except Exception as e:
            logging.error(f"Error loading/generating agent configs: {str(e)}")
            raise
    
    def create_users(self) -> list[AgentUser]:
        """Create users and register them in the database using configs."""
        users = []
        
        try:
            configs = self.load_agent_configs()
            logging.info(f"Creating {len(configs)} users")
            
            for user_config in configs:
                user_id = Utils.generate_formatted_id("user")
                self.db_manager.add_user(user_id, user_config)
                user = AgentUser(
                    user_id=user_id, 
                    user_config=user_config, 
                    temperature=self.experiment_config['temperature'],
                    experiment_config=self.experiment_config
                )
                users.append(user)
            
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]
            logging.info(f"Total users in database after creation: {count}")
            
        except Exception as e:
            logging.error(f"Error creating users: {str(e)}")
            self.conn.rollback()
            raise
        
        return users
        
    def create_initial_follows(self):
        """Create initial follow relationships between users based on Barabási-Albert model.
        
        This implements a preferential attachment model where:
        1. We start with a small initial connected network
        2. New connections are made with probability proportional to node degree
        """
        # Parameters
        m0 = min(5, len(self.users))  # Initial complete network size
        m = min(3, len(self.users) - m0)  # New edges per node
        
        if len(self.users) <= 1:
            logging.warning("Not enough users to create a network")
            return
            
        # Step 1: Create initial connected network with m0 nodes
        initial_users = self.users[:m0]
        for i, user in enumerate(initial_users):
            for j in range(i+1, len(initial_users)):
                other_user = initial_users[j]
                user.follow_user(other_user.user_id)
                other_user.follow_user(user.user_id)
        
        # Step 2: Add remaining nodes with preferential attachment
        if m > 0:  # Only proceed if we have parameters for preferential attachment
            remaining_users = self.users[m0:]
            for new_user in remaining_users:
                # Calculate current degree distribution
                user_degrees = {}
                for user in self.users:
                    # Get number of followers for each user
                    cursor = self.conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM follows WHERE followed_id = ?", (user.user_id,))
                    followers_count = cursor.fetchone()[0]
                    # Add 1 to avoid zero probability for new nodes
                    user_degrees[user.user_id] = followers_count + 1
                
                # Remove users that are already being followed to avoid duplicates
                cursor = self.conn.cursor()
                cursor.execute("SELECT followed_id FROM follows WHERE follower_id = ?", (new_user.user_id,))
                already_following = [row[0] for row in cursor.fetchall()]
                for user_id in already_following:
                    if user_id in user_degrees:
                        del user_degrees[user_id]
                
                # Skip if no available users to follow
                if not user_degrees:
                    continue
                
                # Select m users to follow based on preferential attachment
                total_degree = sum(user_degrees.values())
                available_users = list(user_degrees.keys())
                probabilities = [user_degrees[uid]/total_degree for uid in available_users]
                
                # Select users to follow (without replacement)
                users_to_follow = []
                for _ in range(min(m, len(available_users))):
                    if not available_users:
                        break
                    # Choose based on probability
                    chosen_idx = random.choices(range(len(available_users)), weights=probabilities, k=1)[0]
                    chosen_user_id = available_users.pop(chosen_idx)
                    probabilities.pop(chosen_idx)
                    users_to_follow.append(chosen_user_id)
                
                # Create follow relationships
                for user_id in users_to_follow:
                    new_user.follow_user(user_id)
        
        logging.info(f"Created scale-free network structure using Barabási-Albert model")
        
    def add_random_users(self, num_users_to_add: int = 1, follow_probability: float = 0.0):
        """Add new random users to the simulation."""
        user_configs = self.load_agent_configs()
        new_users = []
        
        for _ in range(num_users_to_add):
            time.sleep(0.1)
            user_config = random.choice(user_configs)
            user_id = Utils.generate_formatted_id("user")
            user = AgentUser(
                user_id=user_id, 
                user_config=user_config, 
                temperature=self.experiment_config['temperature'],
                experiment_config=self.experiment_config
            )
            new_users.append(user)
            
            # Add to database
            self.db_manager.add_user(user_id, user_config)
            logging.info(f"Added new user {user_id} to the simulation")
            
            # Create initial follows for new user
            for existing_user in self.users:
                if random.random() < follow_probability:
                    user.follow_user(existing_user.user_id)
                if random.random() < follow_probability:
                    existing_user.follow_user(user.user_id)
        
        self.users.extend(new_users)
        logging.info(f"Added {num_users_to_add} new users to the simulation")