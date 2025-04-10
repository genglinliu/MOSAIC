import json
import sqlite3
import pandas as pd
from collections import OrderedDict

# Load the user-prolific mapping
def load_user_prolific_mapping():
    # reset the connection
    conn = sqlite3.connect('20250312_162949.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    users_df = pd.DataFrame(users, columns=[column[0] for column in cursor.description])
    users_df['background_labels'] = users_df['background_labels'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    prolific_ids = users_df['background_labels'].apply(lambda x: x['prolific_id'] if isinstance(x, dict) and 'prolific_id' in x else None)
    # print(prolific_ids)
    # close the connection
    conn.close()
    # return dict
    # print(dict(zip(users_df['user_id'], prolific_ids)))
    return dict(zip(users_df['user_id'], prolific_ids))

# Load and update reactions.json
def update_reactions_with_prolific_ids():
    # Load the mapping
    user_prolific_mapping = load_user_prolific_mapping()
    # Load reactions.json
    with open('reactions.json', 'r') as f:
        reactions = json.load(f)
    
    # Add prolific_id to each user's data
    for user_id in reactions:
        if user_id in user_prolific_mapping:
            reactions[user_id]['prolific_id'] = user_prolific_mapping[user_id]
            print(user_id, user_prolific_mapping[user_id])
    
    # Save updated reactions.json
    with open('reactions.json', 'w') as f:
        json.dump(reactions, f, indent=4)
    
    # Save as JSONL with user_id and prolific_id first
    with open('reactions.jsonl', 'w') as f:
        for user_id, user_data in reactions.items():
            # Create new OrderedDict with user_id and prolific_id first
            ordered_data = OrderedDict()
            ordered_data['user_id'] = user_id
            ordered_data['prolific_id'] = user_data.get('prolific_id')
            # Add all other data
            for key, value in user_data.items():
                if key != 'prolific_id':  # Skip prolific_id as we already added it
                    ordered_data[key] = value
            # Write each user as a single line
            f.write(json.dumps(ordered_data) + '\n')

if __name__ == "__main__":
    update_reactions_with_prolific_ids()
