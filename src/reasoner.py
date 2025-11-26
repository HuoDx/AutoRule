from typing import Tuple, List, Dict, Union
from llmapi import LLM

def conv_to_str(conversation: Union[List[Dict[str, str]], str]) -> str:
    if isinstance(conversation, list):
        return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation)
    return conversation if isinstance(conversation, str) else ""

def get_explanation_prompt(
    conversation_a: Union[List[Dict[str, str]], str],
    conversation_b: Union[List[Dict[str, str]], str],
    winner: str,
) -> str:
    conversation_a_str = conv_to_str(conversation_a)
    conversation_b_str = conv_to_str(conversation_b)
    return f"""
[Instruction]
You are tasked with analyzing two conversations between an AI assistant and a user. Based on the content, please provide a detailed explanation of why the user might have preferred the winning conversation.
Please consider aspects such as clarity, coherence, helpfulness, tone, and overall quality.
[Conversation with Assistant A]
{conversation_a_str}
[Conversation with Assistant B]
{conversation_b_str}
[Winning Conversation]: {winner}
[Your Explanation]
"""

def get_explanation_response(
    conversation_a: Union[List[Dict[str, str]], str],
    conversation_b: Union[List[Dict[str, str]], str],
    winner: str,
    *,
    client: LLM,
) -> Tuple[str, str]:
    prompt = get_explanation_prompt(conversation_a, conversation_b, winner)
    # response = client.converse(
    #     modelId="us.deepseek.r1-v1:0",
    #     messages=[{"role": "user", "content": [{"text": prompt}]}],
    #     inferenceConfig={"temperature": 0.6, "maxTokens": 32768},
    # )
    response_text, reasoning = client.generate(prompt)
    # explanation_text = response["output"]["message"]["content"][0]["text"]
    # reasoning = response["output"]["message"]["content"][1]["reasoningContent"]["reasoningText"]
    # return explanation_text, reasoning
    return response_text, reasoning