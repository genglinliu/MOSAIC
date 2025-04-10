# sort the reactions.jsonl file by the user_id

import json

with open('human_reactions.jsonl', 'r') as f:
    reactions = [json.loads(line) for line in f]

# sort the reactions by the user_id
reactions.sort(key=lambda x: x['prolific_id'])

# save the sorted reactions
with open('human_reactions_sorted.jsonl', 'w') as f:
    for reaction in reactions:
        f.write(json.dumps(reaction) + '\n')