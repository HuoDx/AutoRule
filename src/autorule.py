import os
import json
import argparse
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
# import boto3
# from botocore.config import Config

from utils import load_mt_subset, load_ultra_subset, load_hpdv3_subset
# from reasoner import get_explanation_response
from vision_reasoner import get_explanation_response

from extractor import get_extracted_rules
from merger import get_merged_rules
from llmapi import Gemini

load_dotenv()
# config = Config(read_timeout=900, connect_timeout=900)
# bedrock = boto3.client(
#     service_name="bedrock-runtime",
#     region_name="us-east-1",
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     config=config,
# )

# provider = DeepSeek( 
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     model_name="deepseek-reasoner",
#     max_tokens=4096,
#     temperature=0.6,
# )

# provider = OpenAI( 
#     api_key=os.getenv("OPENAI_API_KEY"),
#     model_name="gpt-5-nano",
#     max_tokens=4096,
#     temperature=0.6,
# )

provider = Gemini(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name="gemini-2.5-flash-lite",
    max_tokens=4096,
    temperature=0.6,
)

def evaluate_example(example):
    try:
        # Step 1: explanation generation
        winner = example["winner"]
        explanation_text, reasoning = get_explanation_response(
            example["conversation_a"], example["conversation_b"], winner, client=provider
        )
        # Step 2: rule extraction
        extracted_rules = get_extracted_rules(
            explanation_text, winner.split("_")[1].upper(), client=provider
        )
        return {
            "winner": winner,
            "explanation": explanation_text,
            "reasoning": reasoning,
            "extracted_rules": extracted_rules,
            "conversation_a": example["conversation_a"],
            "conversation_b": example["conversation_b"],
        }
    except Exception as e:
        # Wrap the exception to avoid pickling issues with custom exception classes like APIStatusError
        # We convert the exception to a string to ensure it can be pickled
        raise RuntimeError(f"Error in evaluate_example: {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["uf", "mt", "hpdv3"], default="uf", help="Dataset to evaluate (ultra-feedback or MT-bench)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the results")
    parser.add_argument("--uf_num_examples", type=int, default=256, help="Number of examples to evaluate (for uf)")
    parser.add_argument("--mt_num_examples_per_question", type=int, default=8, help="Number of examples to evaluate per question (for mt)")
    parser.add_argument("--hpdv3_num_examples", type=int, default=None, help="Number of examples to evaluate (for hpdv3)")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes to use for parallel evaluation")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = args.output_dir or f"results_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)

    if args.dataset_type == "uf":
        subset = load_ultra_subset(num_examples=args.uf_num_examples, randomize=True, train=True)
    elif args.dataset_type == "mt":
        subset = load_mt_subset(k=args.mt_num_examples_per_question, train=True)
    elif args.dataset_type == "hpdv3":
        subset = load_hpdv3_subset(num_examples=args.hpdv3_num_examples)
    else:
        raise Exception(f"{args.dataset_type} not found as a dataset")

    mapped_results = subset.map(evaluate_example, num_proc=args.num_proc)
    results = mapped_results.to_list()
    df_explanation = pd.DataFrame(results)
    explanation_tsv = os.path.join(results_folder, "explanation_results.tsv")
    explanation_json = os.path.join(results_folder, "explanation_results.json")
    df_explanation.to_csv(explanation_tsv, sep="\t", index=False)
    df_explanation.to_json(explanation_json, orient="records", indent=4)

    # collect rules
    all_rules = [rule for rules in df_explanation["extracted_rules"] if isinstance(rules, list) for rule in rules]
    unique_rules = list(set(all_rules))

    # Step 3. merge
    merged_rules = get_merged_rules(unique_rules, client=provider)
    merged_rules_filename = os.path.join(results_folder, "merged_rules.json")
    with open(merged_rules_filename, "w", encoding="utf-8") as f:
        json.dump(merged_rules, f, indent=4, ensure_ascii=False)
    print("Merged rules saved to:", merged_rules_filename)

if __name__ == "__main__":
    main()
