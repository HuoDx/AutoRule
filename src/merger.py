import json
from typing import List
from llmapi import LLM

def get_rule_merging_prompt(rules_list: List[str]) -> str:
    rules_text = "\n".join(f"- {rule}" for rule in rules_list)
    return f"""
[Instruction]
Below is a large list of rule-like statements regarding the behavior of an AI assistant. Some of these rules might be duplicates or very similar in meaning.
Please merge them so that there are no duplicates or rules with very similar meanings.
Return the merged list as a JSON array of strings. Do not use ```json```, just output the JSON array directly.
[Rules]
{rules_text}
"""

def get_merged_rules(rules_list: List[str], *, client: LLM) -> List[str]:
    prompt = get_rule_merging_prompt(rules_list)
    # response = client.converse(
    #     modelId="us.deepseek.r1-v1:0",
    #     messages=[{"role": "user", "content": [{"text": prompt}]}],
    #     inferenceConfig={"temperature": 0.6, "maxTokens": 32768},
    # )
    response = client.generate_json(prompt)
    # merged_text = response["output"]["message"]["content"][0]["text"].strip()
    merged_text = response.strip()
    try:
        merged_rules = json.loads(merged_text)
        return merged_rules if isinstance(merged_rules, list) else []
    except Exception:
        return []