#!/usr/bin/env python3
"""
This script merges two JSONL files containing old fake news and new ones from NewsGuard 1219, creating a single file with all unique items.

Usage:
    python get_fake_news.py

"""
import json
import os

def load_jsonl(file_path):
    """Load JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def find_union_and_merge(myths_path, fake_news_path, output_path):
    """Find union between two JSONL files and create a merged file without duplicates."""
    print(f"Loading myths from {myths_path}...")
    myths = load_jsonl(myths_path)
    print(f"Loaded {len(myths)} myths.")
    
    print(f"Loading fake news from {fake_news_path}...")
    fake_news = load_jsonl(fake_news_path)
    print(f"Loaded {len(fake_news)} fake news items.")
    
    # Create dictionaries of names for quick lookup, filtering out items with missing or None names
    myths_names = {}
    for item in myths:
        if item.get('name') is not None:
            myths_names[item['name'].lower()] = item
        else:
            print("Warning: Found a myth item with missing or None 'name' field")
    
    fake_news_names = {}
    for item in fake_news:
        if item.get('name') is not None:
            fake_news_names[item['name'].lower()] = item
        else:
            print("Warning: Found a fake news item with missing or None 'name' field")
    
    print(f"After filtering: {len(myths_names)} myths, {len(fake_news_names)} fake news items")
    
    # Create merged dictionary with all unique items
    merged_dict = myths_names.copy()  # Start with all myths
    
    # Add fake news items that aren't already in myths
    unique_fake_news = 0
    for name, item in fake_news_names.items():
        if name not in merged_dict:
            merged_dict[name] = item
            unique_fake_news += 1
    
    # Convert dictionary to list for output
    merged_data = list(merged_dict.values())
    
    print(f"Union contains {len(merged_data)} items:")
    print(f"- {len(myths_names)} from myths dataset")
    print(f"- {unique_fake_news} unique items from fake news dataset")
    print(f"- {len(fake_news_names) - unique_fake_news} duplicate items (in both datasets)")
    
    print(f"Saving {len(merged_data)} merged items to {output_path}...")
    save_jsonl(merged_data, output_path)
    print("Done!")
    
    # Print some examples from the merged data
    print("\nExamples from the merged data:")
    for i, item in enumerate(merged_data[:5]):
        print(f"{i+1}. {item['name']}")

if __name__ == "__main__":
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    myths_path = os.path.join(current_dir, "myths.jsonl")
    fake_news_path = os.path.join(current_dir, "fake_news.jsonl")
    output_path = os.path.join(current_dir, "merged_union.jsonl")
    
    # Find union and create merged file
    find_union_and_merge(myths_path, fake_news_path, output_path)
