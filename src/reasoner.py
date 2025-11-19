import os
from typing import Tuple, List, Dict, Union

from llm_api import ChatMessage, ContentBlock, InferenceConfig, LLMClient

DEFAULT_MODEL_ID = os.getenv("LLM_MODEL_ID", "us.deepseek.r1-v1:0")
DEFAULT_INFERENCE_CONFIG = InferenceConfig(temperature=0.6, max_tokens=32768)

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
    client: LLMClient,
    model_id: str | None = None,
) -> Tuple[str, str]:
    prompt = get_explanation_prompt(conversation_a, conversation_b, winner)
    response = client.converse(
        model_id=model_id or DEFAULT_MODEL_ID,
        messages=[ChatMessage(role="user", content=[ContentBlock(text=prompt)])],
        inference_config=DEFAULT_INFERENCE_CONFIG,
    )
    explanation_text = response.first_text() or ""
    reasoning = response.first_reasoning() or ""
    return explanation_text, reasoning
