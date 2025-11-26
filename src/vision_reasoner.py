from typing import Tuple, List, Dict, Union
from llmapi import LLM
from utils import ImageFileProvider
from google.genai import types

def conv_to_str(conversation: Union[List[Dict[str, str]], str]) -> str:
    if isinstance(conversation, list):
        return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation)
    return conversation if isinstance(conversation, str) else ""

def get_explanation_prompt(
    conversation_a: Union[List[Dict[str, str]], str],
    conversation_b: Union[List[Dict[str, str]], str],
    winner: str,
) -> str:
    # this version needs to support base64 images, therefore we cannot use the conv_to_str function

    # conversation_a_str = conv_to_str(conversation_a)
    # conversation_b_str = conv_to_str(conversation_b)
    # return f"""
    #     [Instruction]
    #     You are tasked with analyzing two conversations between an AI assistant and a user. Based on the content, please provide a detailed explanation of why the user might have preferred the winning conversation.
    #     Please consider aspects such as clarity, coherence, helpfulness, tone, and overall quality.
    #     [Conversation with Assistant A]
    #     {conversation_a_str}
    #     [Conversation with Assistant B]
    #     {conversation_b_str}
    #     [Winning Conversation]: {winner}
    #     [Your Explanation]
    # """
    # we need to entirely re-pack the above prompt into a correct message object for llm apis
    # Gemini version!
    prompt_a = conversation_a[0]
    image_a = types.Part.from_bytes(
        data=ImageFileProvider.retrieve(conversation_a[1]),
        mime_type="image/jpeg",
    )
    prompt_b = conversation_b[0]
    image_b = types.Part.from_bytes(
        data=ImageFileProvider.retrieve(conversation_b[1]),
        mime_type="image/jpeg",
    )
    messages = [
        f'''
        [Instruction]
        You are tasked with analyzing two conversations between an AI assistant and a user. Based on the content, please provide a detailed explanation of why the user might have preferred the winning conversation.
        Please consider aspects such as clarity, coherence, helpfulness, tone, and overall quality.
        [Conversation with Assistant A]
        {prompt_a}
        ''',
        image_a,
        f'''
        [Conversation with Assistant B]
        {prompt_b}
        ''',
        image_b,
        f'''
        [Winning Conversation]: {winner}
        [Your Explanation]
        '''
    ]
    # messages = []
    # messages.append({"role": "user", "content": '''
    # [Instruction]
    # You are tasked with analyzing two conversations between an AI assistant and a user. Based on the content, please provide a detailed explanation of why the user might have preferred the winning conversation.
    # Please consider aspects such as clarity, coherence, helpfulness, tone, and overall quality.
    # [Conversation with Assistant A]
    # '''})
    # messages.append({"role": "user", "content": [{"type": "text", "text": conversation_a[0]}, {"type": "image", "image": conversation_a[1]}]})
    # messages.append({"role": "user", "content": '''
    # [Conversation with Assistant B]
    # '''})
    # messages.append({"role": "user", "content": [{"type": "text", "text": conversation_b[0]}, {"type": "image", "image": conversation_b[1]}]})
    # messages.append({"role": "user", "content": f"""[Winning Conversation]: {winner}"""})
    # messages.append({"role": "user", "content": """[Your Explanation]"""})
    
    # # version 2: single line version
    # messages2_content = [
    #     # instruction
    #     {'text': '''
    #     [Instruction]
    #     You are tasked with analyzing two conversations between an AI assistant and a user. Based on the content, please provide a detailed explanation of why the user might have preferred the winning conversation.
    #     Please consider aspects such as clarity, coherence, helpfulness, tone, and overall quality.''',
    #     'type': 'input_text'
    #     },
    #     # conversation A
    #     {'text': f"""
    #     [Conversation with Assistant A]
    #     User: {conversation_a[0]}
    #     Assistant: 
    #     """,
    #     'type': 'input_text'
    #     },
    #     {'image': conversation_a[1],
    #     'type': 'input_image'
    #     },
    #     # conversation B
    #     {'text': f"""
    #     [Conversation with Assistant B]
    #     User: {conversation_b[0]}
    #     Assistant: 
    #     """,
    #     'type': 'input_text'
    #     },
    #     {'image': conversation_b[1],
    #     'type': 'input_image'
    #     },
    #     # winning conversation
    #     {'text': f"""
    #     [Winning Conversation]: {winner}
    #     """,
    #     'type': 'input_text'
    #     },
    # ]
    # # Should I use version 1 or version 2?
    return messages

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