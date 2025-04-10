import logging
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3
import json

class HomophilyAnalysis:
    def __init__(self, db_path: str):
        """
        Initialize the homophily analysis with a path to the simulation database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # Debug: Print table schema
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        print("Users table columns:", [col[1] for col in columns])
        
    def calculate_homophily(self):
        """
        Calculate homophily metrics for the network based on all available user attributes
        in background_labels.
        """
        cursor = self.conn.cursor()
        
        # Get all users and their attributes
        cursor.execute("""
            SELECT u1.user_id, u1.background_labels,
                   u2.user_id, u2.background_labels
            FROM follows f
            JOIN users u1 ON f.follower_id = u1.user_id
            JOIN users u2 ON f.followed_id = u2.user_id
        """)
        connections = cursor.fetchall()
        
        total_connections = len(connections)
        if total_connections == 0:
            return {'total_connections': 0}
        
        # Initialize metrics dictionary
        attribute_matches = {}
        attribute_counts = {}
        
        for conn in connections:
            # Parse JSON strings to dictionaries
            labels1 = json.loads(conn[1]) if conn[1] else {}
            labels2 = json.loads(conn[3]) if conn[3] else {}
            
            # Get all unique attributes
            all_attributes = set(labels1.keys()) | set(labels2.keys())
            
            # Compare each attribute
            for attr in all_attributes:
                if attr not in attribute_matches:
                    attribute_matches[attr] = 0
                    attribute_counts[attr] = 0
                
                # Only count if both users have the attribute
                if attr in labels1 and attr in labels2:
                    attribute_counts[attr] += 1
                    if labels1[attr] == labels2[attr]:
                        attribute_matches[attr] += 1
        
        # Calculate homophily for each attribute
        metrics = {'total_connections': total_connections}
        for attr in attribute_matches:
            if attribute_counts[attr] > 0:
                metrics[f'{attr}_homophily'] = attribute_matches[attr] / attribute_counts[attr]
            else:
                metrics[f'{attr}_homophily'] = 0
                
        return metrics

    def calculate_attribute_assortativity(self):
        """
        Calculate network assortativity for all available attributes using networkx.
        """
        G = nx.DiGraph()
        cursor = self.conn.cursor()
        
        # Get all unique attributes from the network
        cursor.execute("SELECT background_labels FROM users")
        all_attributes = set()
        for (background_labels,) in cursor.fetchall():
            if background_labels:
                labels = json.loads(background_labels)
                all_attributes.update(labels.keys())
        
        # Add nodes with all attributes
        cursor.execute("SELECT user_id, background_labels FROM users")
        for user in cursor.fetchall():
            labels = json.loads(user[1]) if user[1] else {}
            # Add node with all possible attributes (None if not present)
            node_attrs = {attr: labels.get(attr) for attr in all_attributes}
            G.add_node(user[0], **node_attrs)

        # Add edges
        cursor.execute("SELECT follower_id, followed_id FROM follows")
        G.add_edges_from(cursor.fetchall())
        
        # Calculate assortativity for each attribute
        assortativity = {}
        for attr in all_attributes:
            try:
                # Remove nodes with None values for this attribute before calculating
                H = G.copy()
                nodes_to_remove = [n for n, attr_dict in H.nodes(data=True) 
                                 if attr_dict.get(attr) is None]
                H.remove_nodes_from(nodes_to_remove)
                
                if H.number_of_nodes() > 1:
                    assortativity[f'{attr}_assortativity'] = nx.attribute_assortativity_coefficient(H, attr)
                else:
                    assortativity[f'{attr}_assortativity'] = None
            except Exception as e:
                logging.warning(f"Could not calculate assortativity for {attr}: {str(e)}")
                assortativity[f'{attr}_assortativity'] = None
        
        return assortativity

    def visualize_homophily_network(self, output_dir: str):
        """
        Create network visualizations for each attribute found in background_labels.
        
        Args:
            output_dir: Directory to save the visualizations
        """
        cursor = self.conn.cursor()
        
        # Get all unique attributes
        cursor.execute("SELECT background_labels FROM users")
        all_attributes = set()
        for (background_labels,) in cursor.fetchall():
            if background_labels:
                labels = json.loads(background_labels)
                all_attributes.update(labels.keys())
        
        # Create a visualization for each attribute
        for attribute in all_attributes:
            G = nx.DiGraph()
            
            # Add nodes with the current attribute
            cursor.execute("SELECT user_id, background_labels FROM users")
            for user in cursor.fetchall():
                labels = json.loads(user[1]) if user[1] else {}
                attribute_value = labels.get(attribute, 'Unknown')
                G.add_node(user[0], attribute_value=attribute_value)
            
            # Add edges
            cursor.execute("SELECT follower_id, followed_id FROM follows")
            G.add_edges_from(cursor.fetchall())
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42, k=1)
            
            # Get unique attribute values for coloring
            attribute_values = list(set(nx.get_node_attributes(G, 'attribute_value').values()))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(attribute_values)))
            color_map = dict(zip(attribute_values, colors))
            
            # Draw nodes colored by attribute
            for value in attribute_values:
                nodes = [n for n, attr in G.nodes(data=True) 
                        if attr['attribute_value'] == value]
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                     node_color=[color_map[value]], 
                                     label=value)
            
            nx.draw_networkx_edges(G, pos, alpha=0.2)
            plt.title(f"Network Colored by {attribute}")
            plt.legend()
            
            # Save visualization
            output_path = Path(output_dir) / f"homophily_{attribute}_network.png"
            plt.savefig(output_path)
            plt.close()

    def run_analysis(self, output_dir: str):
        """
        Run all homophily analyses and save results.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate metrics
        homophily_metrics = self.calculate_homophily()
        assortativity_metrics = self.calculate_attribute_assortativity()
        
        # Log results
        logging.info("\nHomophily Analysis Results:")
        for attr, value in homophily_metrics.items():
            if attr != 'total_connections':
                logging.info(f"{attr}: {value:.3f}")
        
        logging.info("\nNetwork Assortativity:")
        for attr, value in assortativity_metrics.items():
            if value is not None:
                logging.info(f"{attr}: {value:.3f}")
        
        # Create visualizations for all attributes
        self.visualize_homophily_network(output_dir)
        
        # Save metrics to file
        results = {
            'homophily_metrics': homophily_metrics,
            'assortativity_metrics': assortativity_metrics
        }
        
        with open(Path(output_dir) / 'homophily_results.txt', 'w') as f:
            for category, metrics in results.items():
                f.write(f"\n{category.upper()}:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value}\n")

    def analyze_note_interactions(self):
        """Analyze how users with similar backgrounds interact through notes"""
        self.cursor.execute('''
            SELECT 
                u1.background_labels as noter_background,
                u2.background_labels as poster_background,
                COUNT(*) as interaction_count
            FROM community_notes cn
            JOIN users u1 ON cn.author_id = u1.user_id
            JOIN posts p ON cn.post_id = p.post_id
            JOIN users u2 ON p.author_id = u2.user_id
            GROUP BY u1.user_id, u2.user_id
        ''')

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Specify the path to your simulation database
    db_name = "20250121_105825.db"   
    db_path = f"experiment_outputs/database_copies/{db_name}"
    output_dir = f"experiment_outputs/homophily_analysis/{db_name}"
    
    analyzer = HomophilyAnalysis(db_path)
    analyzer.run_analysis(output_dir) 