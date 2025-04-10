import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os
from tabulate import tabulate

# Create results directory if it doesn't exist
os.makedirs('experiment_outputs/prolific_replication/results', exist_ok=True)

# Load the JSON Lines files into pandas DataFrames
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return pd.DataFrame([json.loads(line) for line in f])

# Load reaction data
agent_reactions_1 = load_jsonl('experiment_outputs/prolific_replication/agent_reactions_feed_1.jsonl')
agent_reactions_2 = load_jsonl('experiment_outputs/prolific_replication/agent_reactions_feed_2.jsonl')
human_reactions_1 = load_jsonl('experiment_outputs/prolific_replication/human_reactions_feed_1.jsonl')
human_reactions_2 = load_jsonl('experiment_outputs/prolific_replication/human_reactions_feed_2.jsonl')

# Load demographic data
human_demographics = load_jsonl('experiment_outputs/prolific_replication/human_reactions.jsonl')

print(f"Feed 1 samples: {len(human_reactions_1)} human, {len(agent_reactions_1)} agent")
print(f"Feed 2 samples: {len(human_reactions_2)} human, {len(agent_reactions_2)} agent")
print(f"Total pairs: {len(human_reactions_1) + len(human_reactions_2)}")
print(f"Human demographic entries: {len(human_demographics)}")

# Combine reactions from both feeds
agent_reactions = pd.concat([agent_reactions_1, agent_reactions_2])
human_reactions_1['feed'] = 'feed_1'
human_reactions_2['feed'] = 'feed_2'
human_reactions = pd.concat([human_reactions_1, human_reactions_2])

# Merge reactions with demographic data using prolific_id
human_data = pd.merge(human_reactions, human_demographics, on='prolific_id', how='inner')
agent_data = pd.merge(agent_reactions, human_demographics, on='prolific_id', how='inner')

print(f"Merged human data entries: {len(human_data)}")
print(f"Merged agent data entries: {len(agent_data)}")

# Create raw text file to store statistical results
raw_stats_file = 'experiment_outputs/prolific_replication/results/raw_statistical_results.txt'
with open(raw_stats_file, 'w') as f:
    f.write("Statistical Test Results: Human vs Agent Engagement by Demographic\n")
    f.write("==================================================================\n\n")

# Dictionary to store all statistical results for summary table
all_stats_results = []

def analyze_post_engagement(df):
    """Analyze how many likes/shares/comments each post received"""
    # Initialize counters for each post
    post_engagement = {}
    
    # Go through each user's actions
    for _, row in df.iterrows():
        for action in row['actions']:
            action_type = action.get('action', '')
            target = action.get('target', '')
            
            # Skip if no target post or not a relevant action or "ignore" action
            if not target or not action_type or action_type == 'ignore':
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
    
    # Handle case where there might be no engagement
    if engagement_df.empty:
        return pd.DataFrame(columns=['likes', 'shares', 'comments'])
    
    return engagement_df

def get_significance_symbol(p_value):
    """Convert p-value to significance symbol"""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"

def analyze_by_attribute_value(human_data, agent_data, attribute, value):
    """Analyze engagement patterns for a specific attribute value"""
    # Filter human and agent data by attribute value
    human_filtered = human_data[human_data[attribute] == value]
    agent_filtered = agent_data[agent_data[attribute] == value]
    
    print(f"\n--- Analysis for {attribute}: {value} ---")
    print(f"Number of humans in this group: {len(human_filtered)}")
    print(f"Number of agents in this group: {len(agent_filtered)}")
    
    # Write to raw results file
    with open(raw_stats_file, 'a') as f:
        f.write(f"\n--- Analysis for {attribute}: {value} ---\n")
        f.write(f"Number of humans in this group: {len(human_filtered)}\n")
        f.write(f"Number of agents in this group: {len(agent_filtered)}\n")
    
    # If no humans or agents in this group, skip analysis
    if len(human_filtered) == 0 or len(agent_filtered) == 0:
        print(f"Insufficient data for {attribute} = {value}")
        with open(raw_stats_file, 'a') as f:
            f.write(f"Insufficient data for {attribute} = {value}\n")
        return None
    
    # Analyze engagement for this demographic group
    human_engagement = analyze_post_engagement(human_filtered)
    agent_engagement = analyze_post_engagement(agent_filtered)
    
    # Skip if there's not enough engagement data
    if human_engagement.empty or len(human_engagement.columns) == 0:
        print(f"No engagement data for humans with {attribute} = {value}")
        with open(raw_stats_file, 'a') as f:
            f.write(f"No engagement data for humans with {attribute} = {value}\n")
        return None
    
    if agent_engagement.empty or len(agent_engagement.columns) == 0:
        print(f"No engagement data for agents with {attribute} = {value}")
        with open(raw_stats_file, 'a') as f:
            f.write(f"No engagement data for agents with {attribute} = {value}\n")
        return None
    
    # Calculate averages
    human_means = human_engagement.mean()
    agent_means = agent_engagement.mean()
    
    # Calculate standard errors, handling case where there's only one data point
    human_sems = human_engagement.sem() if len(human_engagement) > 1 else pd.Series(0, index=human_engagement.columns)
    agent_sems = agent_engagement.sem() if len(agent_engagement) > 1 else pd.Series(0, index=agent_engagement.columns)
    
    print("\nAverage engagement per post:")
    with open(raw_stats_file, 'a') as f:
        f.write("\nAverage engagement per post:\n")
        
    for action in ['likes', 'shares', 'comments']:
        if action in human_means:
            print(f"Human average {action}: {human_means[action]:.2f}")
            with open(raw_stats_file, 'a') as f:
                f.write(f"Human average {action}: {human_means[action]:.2f}\n")
        if action in agent_means:
            print(f"Agent average {action}: {agent_means[action]:.2f}")
            with open(raw_stats_file, 'a') as f:
                f.write(f"Agent average {action}: {agent_means[action]:.2f}\n")
    
    # Statistical comparison
    print("\nStatistical Comparison (t-test):")
    with open(raw_stats_file, 'a') as f:
        f.write("\nStatistical Comparison (t-test):\n")
    
    # Store t-test results for this group
    stat_results = {
        'attribute': attribute,
        'value': value,
        'n_humans': len(human_filtered),
        'n_agents': len(agent_filtered)
    }
    
    for action in ['likes', 'shares', 'comments']:
        if action in human_engagement.columns and action in agent_engagement.columns:
            if len(human_engagement[action].dropna()) > 1 and len(agent_engagement[action].dropna()) > 1:
                t_stat, p_val = stats.ttest_ind(
                    human_engagement[action].dropna(),
                    agent_engagement[action].dropna(),
                    equal_var=False  # Use Welch's t-test for unequal variances
                )
                sig_symbol = get_significance_symbol(p_val)
                
                print(f"{action.capitalize()}: t={t_stat:.4f}, p={p_val:.4f} {sig_symbol}")
                with open(raw_stats_file, 'a') as f:
                    f.write(f"{action.capitalize()}: t={t_stat:.4f}, p={p_val:.4f} {sig_symbol}\n")
                
                # Store in results dictionary
                stat_results[f'{action}_t'] = t_stat
                stat_results[f'{action}_p'] = p_val
                stat_results[f'{action}_sig'] = sig_symbol
                stat_results[f'{action}_human_mean'] = human_means.get(action, 0)
                stat_results[f'{action}_agent_mean'] = agent_means.get(action, 0)
                
            else:
                print(f"{action.capitalize()}: Insufficient data for statistical test")
                with open(raw_stats_file, 'a') as f:
                    f.write(f"{action.capitalize()}: Insufficient data for statistical test\n")
                
                # Store as missing
                stat_results[f'{action}_t'] = np.nan
                stat_results[f'{action}_p'] = np.nan
                stat_results[f'{action}_sig'] = "n/a"
                stat_results[f'{action}_human_mean'] = human_means.get(action, 0)
                stat_results[f'{action}_agent_mean'] = agent_means.get(action, 0)
        else:
            print(f"{action.capitalize()}: Missing data")
            with open(raw_stats_file, 'a') as f:
                f.write(f"{action.capitalize()}: Missing data\n")
            
            # Store as missing
            stat_results[f'{action}_t'] = np.nan
            stat_results[f'{action}_p'] = np.nan
            stat_results[f'{action}_sig'] = "n/a"
            stat_results[f'{action}_human_mean'] = human_means.get(action, 0)
            stat_results[f'{action}_agent_mean'] = agent_means.get(action, 0)
    
    # Add this group's results to the overall results list
    all_stats_results.append(stat_results)
    
    # Create visualization
    visualize_comparison(human_means, agent_means, human_sems, agent_sems, attribute, value)
    
    return stat_results

def visualize_comparison(human_means, agent_means, human_sems, agent_sems, attribute, value):
    """Create a bar chart comparing human vs agent engagement for a specific attribute value"""
    plt.figure(figsize=(8, 6))
    x = range(3)  # likes, shares, comments
    width = 0.35
    FONT_SIZE = 16
    
    # Get values for plotting, defaulting to 0 if not available
    human_values = [human_means.get('likes', 0), human_means.get('shares', 0), human_means.get('comments', 0)]
    agent_values = [agent_means.get('likes', 0), agent_means.get('shares', 0), agent_means.get('comments', 0)]
    human_errors = [human_sems.get('likes', 0), human_sems.get('shares', 0), human_sems.get('comments', 0)]
    agent_errors = [agent_sems.get('likes', 0), agent_sems.get('shares', 0), agent_sems.get('comments', 0)]
    
    # Create bars with error bars
    plt.bar([i - width/2 for i in x], human_values, width, label='Human', color='#2ecc71', alpha=0.8,
            yerr=human_errors, capsize=5, ecolor='black', error_kw={'elinewidth': 1})
    
    plt.bar([i + width/2 for i in x], agent_values, width, label='Agent', color='#3498db', alpha=0.8,
            yerr=agent_errors, capsize=5, ecolor='black', error_kw={'elinewidth': 1})
    
    # Add labels and styling
    plt.title(f'Engagement: {attribute} = {value}', fontsize=FONT_SIZE, pad=20)
    plt.xlabel('Engagement Type', fontsize=FONT_SIZE)
    plt.ylabel('Average Number per Post', fontsize=FONT_SIZE)
    plt.xticks(x, ['Likes', 'Shares', 'Comments'], rotation=0, fontsize=FONT_SIZE-2)
    plt.legend(fontsize=FONT_SIZE-2)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels
    for i in x:
        plt.text(i - width/2, human_values[i] + max(human_errors[i], 0.05), f'{human_values[i]:.2f}', 
                ha='center', va='bottom', fontsize=FONT_SIZE-4)
        plt.text(i + width/2, agent_values[i] + max(agent_errors[i], 0.05), f'{agent_values[i]:.2f}', 
                ha='center', va='bottom', fontsize=FONT_SIZE-4)
    
    plt.tight_layout()
    
    # Save the figure - create a safe filename
    safe_value = str(value).replace('/', '_').replace(' ', '_').replace('\\', '_')
    safe_value = ''.join(c for c in safe_value if c.isalnum() or c in '_-')
    plt.savefig(f'experiment_outputs/prolific_replication/results/engagement_{attribute}_{safe_value}.pdf', 
                dpi=300, bbox_inches='tight')
    plt.close()

# Attributes to analyze
attributes_to_analyze = [
    'age', 'gender', 'religion', 'ethnic_group', 'education', 'income', 'political_stance'
]

# Analyze each attribute
for attribute in attributes_to_analyze:
    if attribute not in human_demographics.columns:
        print(f"\n=== Skipping attribute: {attribute} (not found in data) ===")
        continue
        
    print(f"\n=== Analysis for attribute: {attribute} ===")
    with open(raw_stats_file, 'a') as f:
        f.write(f"\n\n=== Analysis for attribute: {attribute} ===\n")
    
    # Get all unique values for this attribute
    unique_values = human_demographics[attribute].dropna().unique()
    print(f"Found {len(unique_values)} unique values: {unique_values}")
    with open(raw_stats_file, 'a') as f:
        f.write(f"Found {len(unique_values)} unique values: {unique_values}\n")
    
    # Analyze each value
    for value in unique_values:
        analyze_by_attribute_value(human_data, agent_data, attribute, value)

# Create summary table as DataFrame
if all_stats_results:
    df_summary = pd.DataFrame(all_stats_results)
    
    # Round numerical columns to 3 decimal places
    numeric_columns = df_summary.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_columns:
        df_summary[col] = df_summary[col].round(3)
    
    # Save full results to CSV
    df_summary.to_csv('experiment_outputs/prolific_replication/results/statistical_results_full.csv', index=False)
    
    # Create a more readable summary table
    summary_table = []
    for _, row in df_summary.iterrows():
        # Count significant engagement types
        significant_count = 0
        for action in ['likes', 'shares', 'comments']:
            sig = row.get(f'{action}_sig', 'n/a')
            if sig in ['*', '**', '***']:
                significant_count += 1
        
        # Determine overall significance based on 1+ criterion
        overall_significance = "Significant" if significant_count >= 1 else "Not Significant"
        
        summary_row = [
            row['attribute'],
            row['value'],
            row['n_humans'],
            row['n_agents'],
            f"{row['likes_human_mean']:.2f} vs {row['likes_agent_mean']:.2f}",
            f"{row['likes_t']:.2f} ({row['likes_sig']})",
            f"{row['shares_human_mean']:.2f} vs {row['shares_agent_mean']:.2f}",
            f"{row['shares_t']:.2f} ({row['shares_sig']})",
            f"{row['comments_human_mean']:.2f} vs {row['comments_agent_mean']:.2f}",
            f"{row['comments_t']:.2f} ({row['comments_sig']})",
            overall_significance
        ]
        summary_table.append(summary_row)
    
    # Create readable table headers
    headers = [
        "Attribute", "Value", "n (Human)", "n (Agent)",
        "Likes (Human vs Agent)", "Likes t-stat (sig)", 
        "Shares (Human vs Agent)", "Shares t-stat (sig)", 
        "Comments (Human vs Agent)", "Comments t-stat (sig)",
        "Overall Significance"
    ]
    
    # Save formatted table to text file
    with open('experiment_outputs/prolific_replication/results/statistical_results_summary.txt', 'w') as f:
        f.write("Statistical Test Results Summary: Human vs Agent Engagement by Demographic\n")
        f.write("=====================================================================\n\n")
        f.write(tabulate(summary_table, headers=headers, tablefmt="grid"))
        
        # Add significance legend
        f.write("\n\nSignificance Legend:\n")
        f.write("*** p < 0.001 (highly significant)\n")
        f.write("**  p < 0.01 (very significant)\n")
        f.write("*   p < 0.05 (significant)\n")
        f.write("ns  p ≥ 0.05 (not significant)\n")
        f.write("n/a insufficient data\n\n")
        f.write("Overall Significance: A demographic group is considered 'Significant' if TWO OR MORE\n")
        f.write("engagement types (likes, shares, comments) show statistically significant differences (p < 0.05).\n")
    
    # Generate summary of significant findings using >= 1 criteria
    groups_significant = set()     # Groups with 1+ engagement types showing significance
    groups_nonsignificant = set()  # Groups with fewer than 1 significant engagement types
    
    for _, row in df_summary.iterrows():
        attr_value = f"{row['attribute']} = {row['value']}"
        
        # Count how many engagement types are significant for this group
        significant_count = 0
        for action in ['likes', 'shares', 'comments']:
            sig = row.get(f'{action}_sig', 'n/a')
            if sig in ['*', '**', '***']:
                significant_count += 1
        
        # Apply criteria: significant if 1 or more types show significance
        if significant_count >= 1:  # One or more types are significant
            groups_significant.add(attr_value)
        else:  # Fewer than one significant types
            groups_nonsignificant.add(attr_value)
    
    # Generate detailed findings for the summary
    significant_findings = []
    nonsignificant_findings = []
    
    for _, row in df_summary.iterrows():
        attr_value = f"{row['attribute']} = {row['value']}"
        
        # Get list of which engagements are significant
        sig_actions = []
        nonsig_actions = []
        for action in ['likes', 'shares', 'comments']:
            sig = row.get(f'{action}_sig', 'n/a')
            if sig in ['*', '**', '***']:
                sig_actions.append(action)
            elif sig == 'ns':
                nonsig_actions.append(action)
        
        # Generate detailed findings based on the categorization
        if attr_value in groups_significant:
            significant_findings.append(f"{attr_value} shows significant differences in {', '.join(sig_actions)}" + 
                                      (f" (non-significant in {', '.join(nonsig_actions)})" if nonsig_actions else ""))
        else:
            if sig_actions:
                nonsignificant_findings.append(f"{attr_value} shows significance only in {', '.join(sig_actions)}")
            else:
                nonsignificant_findings.append(f"{attr_value} shows no significant differences in any engagement type")
    
    # Save summary to text file
    with open('experiment_outputs/prolific_replication/results/significant_findings_summary.txt', 'w') as f:
        f.write("Summary of Differences in Human vs Agent Engagement Patterns\n")
        f.write("=================================================================\n\n")
        f.write("Using criteria: A demographic group is considered 'significantly different'\n")
        f.write("if TWO OR MORE engagement types (likes, shares, comments) show significant differences (p < 0.05).\n\n")
        
        f.write("1. Demographic Groups with SIGNIFICANT Differences (1+ types):\n")
        f.write("----------------------------------------------------------------\n")
        if significant_findings:
            for finding in significant_findings:
                f.write(f"- {finding}\n")
        else:
            f.write("No groups showed significant differences in two or more engagement types.\n")
        
        f.write("\n\n2. Demographic Groups with NON-SIGNIFICANT Results (0-1 types):\n")
        f.write("-----------------------------------------------\n")
        if nonsignificant_findings:
            for finding in nonsignificant_findings:
                f.write(f"- {finding}\n")
        else:
            f.write("All tested demographic groups showed significant differences in at least one engagement type.\n")
        
        # Add overall summary with clear definitions
        f.write("\n\nOverall Summary:\n")
        f.write("--------------\n")
        f.write(f"Total demographic groups analyzed: {len(df_summary)}\n")
        f.write(f"Groups with SIGNIFICANT differences (1+ types): {len(groups_significant)}\n")
        f.write(f"Groups with NON-SIGNIFICANT differences (0-1 types): {len(groups_nonsignificant)}\n")
        
        # Provide explicit explanation of criteria
        f.write("\n\nCriteria for Significance:\n")
        f.write("----------------------\n")
        f.write("- 'Significant difference': p < 0.05 in statistical comparison between human and agent engagement\n")
        f.write("- A demographic group is considered 'significantly different' if TWO OR MORE engagement types\n")
        f.write("  (out of likes, shares, and comments) show statistically significant differences\n")
        f.write("- Otherwise, the group is considered 'non-significant' if it has 0-1 significant engagement types\n")
    
    print("\nAnalysis complete. Results saved to experiment_outputs/prolific_replication/results/")
    print("Summary tables and significant findings available in the results directory.")
else:
    print("\nNo statistical results were generated. Check for errors in the analysis.")
