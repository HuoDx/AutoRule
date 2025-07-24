from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hf_username", type=str, required=True, help="Your Hugging Face username")
args = parser.parse_args()

# Load only the specified splits as a DatasetDict
dataset = load_dataset(
    "HuggingFaceH4/ultrafeedback_binarized",
    split={
        "train_prefs": "train_prefs",
        "test_prefs": "test_prefs",
        "train_sft": "train_sft",
        "test_sft": "test_sft"
    }
)

# Define a function to extract the second element of the "messages" list
def extract_response(example):
    if isinstance(example["messages"], list) and len(example["messages"]) > 1:
        example["response"] = example["messages"][1]["content"]
    else:
        example["response"] = None  # Handle edge case
    if isinstance(example["chosen"], list) and len(example["chosen"]) > 1:
        example["chosen_content"] = example["chosen"][1]["content"]
    if isinstance(example["rejected"], list) and len(example["rejected"]) > 1:
        example["rejected_content"] = example["rejected"][1]["content"]
    return example

# Apply the function to every split in the DatasetDict
processed_dataset = dataset.map(extract_response)

# Push the processed dataset to the Hub
processed_dataset.push_to_hub(f"{args.hf_username}/processed_ultrafeedback_binarized_prefs_sft")
