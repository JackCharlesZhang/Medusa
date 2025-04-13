from datasets import load_dataset
import os
import json

# Load the dataset
ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")

# Define output directory
output_dir = "sharegpt/raw"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Split the train set into train and validation
split_data = ds["train"].train_test_split(test_size=0.05, seed=42)

# Save the splits to JSON files
split_data["train"].to_json(f"{output_dir}/train.json")
split_data["test"].to_json(f"{output_dir}/val.json")

print(f"Data successfully split and saved to {output_dir}/train.json and {output_dir}/val.json")
print(f"Train split size: {len(split_data['train'])}")
print(f"Validation split size: {len(split_data['test'])}")