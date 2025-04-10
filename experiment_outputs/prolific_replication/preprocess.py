import json

def normalize_actions(data, feed_key):
    """Normalize the actions format to be consistent"""
    feed_data = data[feed_key]
    
    # If the feed data is a list of objects with 'actions'
    if isinstance(feed_data, list):
        # Flatten all actions into a single list
        all_actions = []
        for item in feed_data:
            if 'actions' in item:
                all_actions.extend(item['actions'])
        return all_actions
    
    # If the feed data is an object with 'actions'
    elif isinstance(feed_data, dict) and 'actions' in feed_data:
        return feed_data['actions']
    
    return []  # Return empty list for any other unexpected format

# Read the original file and create two new files
with open('experiment_outputs/prolific_replication/human_reactions.jsonl', 'r') as f:
    # Open output files
    with open('experiment_outputs/prolific_replication/human_reactions_feed_1.jsonl', 'w') as f1, \
         open('experiment_outputs/prolific_replication/human_reactions_feed_2.jsonl', 'w') as f2:
        
        # Process each line
        for line in f:
            data = json.loads(line.strip())
            
            # Create normalized records
            feed1_record = {
                "prolific_id": data["prolific_id"],
                "actions": normalize_actions(data, "social_feed_1")
            }
            
            feed2_record = {
                "prolific_id": data["prolific_id"],
                "actions": normalize_actions(data, "social_feed_2")
            }
            
            # Write to respective files
            f1.write(json.dumps(feed1_record) + '\n')
            f2.write(json.dumps(feed2_record) + '\n')
