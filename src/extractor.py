import json
import os
from typing import List

from llm_api import ChatMessage, ContentBlock, InferenceConfig, LLMClient

DEFAULT_MODEL_ID = os.getenv("LLM_MODEL_ID", "us.deepseek.r1-v1:0")
DEFAULT_INFERENCE_CONFIG = InferenceConfig(temperature=0.6, max_tokens=32768)

def get_rule_extraction_prompt(reasoning_text: str, winner: str) -> str:
    few_shot = """
Example 1:
- The assistant's responses should validate any assumptions made with sufficient context and examples.
Example 2:
- The assistant's responses should not simply restate information provided by the user as its answer.
Example 3:
- The assistant's responses should have a structure that satisfies the user's request.
"""
    return f"""
[Instruction]
Based on the following reasoning about why conversation with assistant {winner} is better, extract any rule-like statements implied by the reasoning that indicate this preference. Rule-like statements should be able to be judged objectively and deterministically. Below are a few examples of rule-like statements:
{few_shot}
Return the list as a JSON array of strings. Do not use ```json```, just output the JSON array directly. If there are no rule-like statements, return an empty JSON array.
[Reasoning]
{reasoning_text}
"""

def get_extracted_rules(
    reasoning_text: str,
    winner: str,
    *,
    client: LLMClient,
    model_id: str | None = None,
) -> List[str] | None:
    prompt = get_rule_extraction_prompt(reasoning_text, winner)
    response = client.converse(
        model_id=model_id or DEFAULT_MODEL_ID,
        messages=[ChatMessage(role="user", content=[ContentBlock(text=prompt)])],
        inference_config=DEFAULT_INFERENCE_CONFIG,
    )
    extracted_text = (response.first_text() or "").strip()
    try:
        rules = json.loads(extracted_text)
        return rules if isinstance(rules, list) else None
    except Exception:
        return None
