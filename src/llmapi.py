'''
The LLM API that abstracts away specific LLM providers
'''
from typing import List, Dict, Tuple, Protocol
import openai
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from utils import load_hpdv3_subset, ImageFileProvider

class LLM(Protocol):
    # generate a response given a messages object, returning a response and a reasoning
    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        ...
    # generate a JSON response given a messages object
    def generate_json(self, messages: List[Dict[str, str]]) -> str:
        ...

# class OpenAI(LLM):
#     def __init__(self,
#                  api_key: str,  
#                  model_name: str,
#                  base_url: str = "https://api.openai.com",
#                  max_tokens: int = 32768,
#                  temperature: float = 0.6):
#         self.api_key = api_key
#         self.model_name = model_name
#         self.max_tokens = max_tokens
#         self.temperature = temperature
#         # create openai client
#         self.client = openai.OpenAI(
#             api_key=api_key,
#             base_url=base_url,
#         )

#     def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
#         # use responses from reasoning
#         # https://platform.openai.com/docs/guides/reasoning/advice-on-prompting
#         response = self.client.responses.create(
#             model=self.model_name,
#             input=messages,
#             reasoning={"effort": "medium", "summary": "detailed"},
#             max_output_tokens=self.max_tokens,
#             # temperature=self.temperature,
#         )
#         import pprint 
#         pprint.pprint(response)
        
#         content = ""
#         reasoning = ""
        
#         # scan for type reasoning and type message
#         for output_block in response.output:
#             if output_block.type == "message":
#                 content += output_block.content
#             elif output_block.type == "reasoning":
#                 reasoning += output_block.summary[0].text
                
#         return (
#             content, 
#             reasoning
#         )
        
#     def generate_json(self, messages: List[Dict[str, str]]) -> str:
#         # Fallback to chat.completions for JSON if needed, or check if responses supports it.
#         # For now, assuming chat.completions is safer for JSON mode on o1 unless specified otherwise.
#         # However, o1-preview/mini often don't support system messages or some features in chat.completions.
#         # But let's stick to the plan: keep generate_json as is or slight fix.
#         # The original code used chat.completions.create. Let's keep it but ensure it works.
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=messages,
#             max_completion_tokens=self.max_tokens, # o1 uses max_completion_tokens
#             reasoning_effort="medium", # o1 supports this in chat.completions too
#             # temperature=self.temperature, # o1 often fixes temp to 1
#             response_format={"type": "json_object"}
#         )
#         return response.choices[0].message.content

# class DeepSeek(OpenAI):
#     # OpenAI, with base_url "https://api.deepseek.com"
#     def __init__(self,
#                  api_key: str,
#                  model_name: str,
#                  max_tokens: int = 32768,
#                  temperature: float = 0.6):
#         super().__init__(
#             api_key=api_key,
#             base_url="https://api.deepseek.com",
#             model_name=model_name,
#             max_tokens=max_tokens,
#             temperature=temperature,
#         )

    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        # DeepSeek uses standard chat.completions
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        
        content = response.choices[0].message.content
        # DeepSeek R1 often puts reasoning in reasoning_content
        reasoning = getattr(response.choices[0].message, 'reasoning_content', "")
        
        return (
            content, 
            reasoning
        )

# if it's called from main, run a quick sanity test

# hard lesson: vibe coding ain't gonna work :/

class Gemini(LLM):
    def __init__(self, api_key, model_name, max_tokens, temperature):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = genai.Client(api_key=api_key)
    
    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        response = self.client.models.generate_content(
            model=self.model_name, 
            contents=messages,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=-1,
                ),
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        )
        thoughts = ""
        final_answer = ""
        
        # Check if candidates exist
        if not response.candidates:
            return "", ""
            
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            # Check for thought attribute (google-genai SDK)
            if hasattr(part, 'thought') and part.thought:
                thoughts += part.text
            # Check for text attribute
            else:
                final_answer += part.text
                
        return (
            final_answer, 
            thoughts
        )
        
    def generate_json(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.models.generate_content(
            model=self.model_name, 
            contents=messages,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        if response.candidates and response.candidates[0].content.parts:
             return response.candidates[0].content.parts[0].text
        return ""

if __name__ == "__main__":
    load_dotenv()
    llm = Gemini(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.5-flash-lite",
        max_tokens=4096,
        temperature=0.6,
    )

    hpdv3 = load_hpdv3_subset(num_examples=1)
    example = hpdv3[0]
    prompt = [
        "here are two conversations A and B, each containing an image generation task and the result:\n", 
        "\nConversataion A\n",
        'User: ' + example["conversation_a"][0], 
        'Assistant: ',
        types.Part.from_bytes(
            data=ImageFileProvider.retrieve(example["conversation_a"][1]),
            mime_type="image/jpeg",
        ), 
        "\nConversataion B\n",
        'User: ' + example["conversation_b"][0], 
        'Assistant: ',
        types.Part.from_bytes(
            data=ImageFileProvider.retrieve(example["conversation_b"][1]),
            mime_type="image/jpeg",
        ), 
        '\n The winner is: ' + example["winner"] + '\n',
        'Pay attention to both the text instructions and the images, please explain why the winner is better than the other',
    ]
    print(prompt)
    response, reasoning = llm.generate(prompt)
    print("Response:", response)
    print("Reasoning:", reasoning)
