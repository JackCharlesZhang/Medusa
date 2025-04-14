from datasets import load_dataset
import os
import json

# Load the dataset
ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")

# Define output directory
output_dir = "sharegpt/processed"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to convert from ShareGPT format to the proper format
def convert_format(example):
    conversations = []
    original_conversations = example["conversations"]
    
    # Check if the conversation starts with a human message
    valid_conversation = False
    current_role = None
    
    # Process each message
    for msg in original_conversations:
        if "from" in msg and "value" in msg:
            # Map roles
            role = "user" if msg["from"].lower() == "human" else "assistant"
            
            # For the first message, ensure it's from a user
            if not conversations:
                if role != "user":
                    continue  # Skip this conversation if it doesn't start with user
                valid_conversation = True
            
            # Ensure alternation of roles
            if current_role == role:
                continue  # Skip consecutive messages from the same role
                
            # Add message with the correct format
            if msg["value"] and msg["value"].strip():  # Skip empty messages
                conversations.append({
                    "role": role,
                    "content": msg["value"]
                })
                current_role = role
    
    # Only return conversations that have at least one user and one assistant message
    if valid_conversation and len(conversations) >= 2:
        return {"conversations": conversations}
    else:
        return None

# Split the train set into train and validation
split_data = ds["train"].train_test_split(test_size=0.05, seed=42)

# Process and save train data
train_data = []
for item in split_data["train"]:
    processed = convert_format(item)
    if processed:
        train_data.append(processed)

# Process and save validation data
val_data = []
for item in split_data["test"]:
    processed = convert_format(item)
    if processed:
        val_data.append(processed)

# Write to jsonl files (line-delimited JSON)
with open(f"{output_dir}/train.json", "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open(f"{output_dir}/val.json", "w", encoding="utf-8") as f:
    for item in val_data:
        f.write(json.dumps(item) + "\n")

print(f"Data successfully processed and saved to {output_dir}/train.json and {output_dir}/val.json")
print(f"Train split size: {len(train_data)}")
print(f"Validation split size: {len(val_data)}")