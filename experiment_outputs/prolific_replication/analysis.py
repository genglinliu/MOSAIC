import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Load the JSON Lines files into pandas DataFrames
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return pd.DataFrame([json.loads(line) for line in f])


# Load all datasets
agent_reactions_1 = load_jsonl('experiment_outputs/prolific_replication/agent_reactions_feed_1.jsonl')
agent_reactions_2 = load_jsonl('experiment_outputs/prolific_replication/agent_reactions_feed_2.jsonl')
human_reactions_1 = load_jsonl('experiment_outputs/prolific_replication/human_reactions_feed_1.jsonl')
human_reactions_2 = load_jsonl('experiment_outputs/prolific_replication/human_reactions_feed_2.jsonl')

# Combine datasets
agent_reactions = pd.concat([agent_reactions_1, agent_reactions_2])
human_reactions = pd.concat([human_reactions_1, human_reactions_2])

# After loading the data, add:
print(f"Feed 1 samples: {len(human_reactions_1)} human, {len(agent_reactions_1)} agent")
print(f"Feed 2 samples: {len(human_reactions_2)} human, {len(agent_reactions_2)} agent")
print(f"Total pairs: {len(human_reactions_1) + len(human_reactions_2)}")

def analyze_post_engagement(df):
    """Analyze how many likes/shares/comments each post received"""
    # Initialize counters for each post
    post_engagement = {}
    
    # Go through each user's actions
    for _, row in df.iterrows():
        for action in row['actions']:
            action_type = action.get('action', '')
            target = action.get('target', '')
            
            # Skip if no target post or not a relevant action
            if not target or not action_type:
                continue
                
            # Initialize post counters if we haven't seen this post before
            if target not in post_engagement:
                post_engagement[target] = {'likes': 0, 'shares': 0, 'comments': 0}
            
            # Count the engagement
            if action_type.startswith('like'):
                post_engagement[target]['likes'] += 1
            elif action_type.startswith('share'):
                post_engagement[target]['shares'] += 1
            elif action_type.startswith('comment'):
                post_engagement[target]['comments'] += 1
    
    # Convert to DataFrame for easier analysis
    engagement_df = pd.DataFrame.from_dict(post_engagement, orient='index')
    
    # Calculate average engagement per post
    avg_engagement = engagement_df.mean()
    std_engagement = engagement_df.std()
    
    print("\nAverage engagement per post:")
    for action in ['likes', 'shares', 'comments']:
        print(f"Average {action}: {avg_engagement[action]:.2f} (±{std_engagement[action]:.2f})")
    
    return engagement_df

# Analyze human and agent engagement separately
print("\nHUMAN POSTS RECEIVED:")
human_engagement = analyze_post_engagement(human_reactions)

print("\nAGENT POSTS RECEIVED:")
agent_engagement = analyze_post_engagement(agent_reactions)

# Debug print to see target formats
print("\nExample targets from human posts:")
print(list(human_engagement.index)[:5])
print("\nExample targets from agent posts:")
print(list(agent_engagement.index)[:5])

# Visualize the comparison
plt.figure(figsize=(8, 6))  # More square dimensions
x = range(3)  # likes, shares, comments
width = 0.35

# Get the means and standard errors for plotting
human_means = human_engagement.mean()
agent_means = agent_engagement.mean()
human_sems = human_engagement.sem()  # Standard error of the mean
agent_sems = agent_engagement.sem()

# set the fontsize of the plot
FONT_SIZE = 18

# Create bars with error bars
plt.bar([i - width/2 for i in x], human_means, width, label='Human', color='#2ecc71', alpha=0.8,
        yerr=human_sems, capsize=5, ecolor='black', error_kw={'elinewidth': 1})
plt.bar([i + width/2 for i in x], agent_means, width, label='Agent', color='#3498db', alpha=0.8,
        yerr=agent_sems, capsize=5, ecolor='black', error_kw={'elinewidth': 1})

# Add labels and styling
# plt.title('Average Engagement Received per Post: Humans vs Agents', fontsize=14, pad=20)
plt.xlabel('Engagement Type', fontsize=FONT_SIZE)
plt.ylabel('Average Number per Post', fontsize=FONT_SIZE)
plt.xticks(x, ['Likes', 'Shares', 'Comments'], rotation=0, fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# x-axis fontsize
plt.gca().tick_params(axis='x', labelsize=FONT_SIZE-4)
# y-axis fontsize
plt.gca().tick_params(axis='y', labelsize=FONT_SIZE-4)

# Add value labels to the left of each bar
for i in x:
    # For human bars (left bars)
    plt.text(i - width/2 + 0.18, human_means[i], f'{human_means[i]:.2f}', 
             ha='right', va='bottom', fontsize=FONT_SIZE-5)
    # For agent bars (right bars)
    plt.text(i + width/2 + 0.18, agent_means[i], f'{agent_means[i]:.2f}', 
             ha='right', va='bottom', fontsize=FONT_SIZE-5)

plt.tight_layout()
plt.savefig('experiment_outputs/prolific_replication/results/post_engagement_comparison.pdf', dpi=300, bbox_inches='tight')

# Statistical comparison
print("\nStatistical Comparison (t-test):")
for action in ['likes', 'shares', 'comments']:
    t_stat, p_val = stats.ttest_ind(
        human_engagement[action].dropna(),
        agent_engagement[action].dropna()
    )
    print(f"\n{action.capitalize()}:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_val:.4f}")