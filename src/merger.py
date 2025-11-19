import json
import os
from typing import List

from llm_api import ChatMessage, ContentBlock, InferenceConfig, LLMClient

DEFAULT_MODEL_ID = os.getenv("LLM_MODEL_ID", "us.deepseek.r1-v1:0")
DEFAULT_INFERENCE_CONFIG = InferenceConfig(temperature=0.6, max_tokens=32768)

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

def get_merged_rules(
    rules_list: List[str],
    *,
    client: LLMClient,
    model_id: str | None = None,
) -> List[str]:
    prompt = get_rule_merging_prompt(rules_list)
    response = client.converse(
        model_id=model_id or DEFAULT_MODEL_ID,
        messages=[ChatMessage(role="user", content=[ContentBlock(text=prompt)])],
        inference_config=DEFAULT_INFERENCE_CONFIG,
    )
    merged_text = (response.first_text() or "").strip()
    try:
        merged_rules = json.loads(merged_text)
        return merged_rules if isinstance(merged_rules, list) else []
    except Exception:
        return []
