import json
from typing import List
from llmapi import LLM

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

def get_extracted_rules(reasoning_text: str, winner: str, *, client: LLM) -> List[str] | None:
    prompt = get_rule_extraction_prompt(reasoning_text, winner)
    # response = client.converse(
    #     modelId="us.deepseek.r1-v1:0",
    #     messages=[{"role": "user", "content": [{"text": prompt}]}],
    #     inferenceConfig={"temperature": 0.6, "maxTokens": 32768},
    # )
    response = client.generate_json(prompt)
    # extracted_text = response["output"]["message"]["content"][0]["text"].strip()
    extracted_text = response.strip()
    try:
        rules = json.loads(extracted_text)
        return rules if isinstance(rules, list) else None
    except Exception:
        return None