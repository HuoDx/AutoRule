from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

__all__ = [
    "ChatMessage",
    "ContentBlock",
    "ConverseResponse",
    "InferenceConfig",
    "LLMClient",
    "BedrockLLMClient",
    "GeminiLLMClient",
    "create_llm_client",
]

try:  # Optional dependency for Bedrock defaults.
    from botocore.config import Config
except Exception:  # pragma: no cover - botocore not available.
    Config = None


@dataclass
class ContentBlock:
    """A normalized chunk of content exchanged during a conversation."""

    text: str = ""
    type: str = "text"
    metadata: Dict[str, Any] | None = None

    def to_bedrock_dict(self) -> Dict[str, Any]:
        if self.type == "text":
            return {"text": self.text}
        if self.type == "reasoning":
            return {"reasoningContent": {"reasoningText": self.text}}
        payload: Dict[str, Any] = {}
        if self.metadata:
            payload.update(self.metadata)
        if "text" not in payload and self.text:
            payload["text"] = self.text
        return payload


@dataclass
class ChatMessage:
    """A normalized chat message."""

    role: str
    content: List[ContentBlock] = field(default_factory=list)

    def ensure_content(self) -> List[ContentBlock]:
        if not self.content:
            self.content = [ContentBlock(text="")]
        return self.content


@dataclass
class ConverseResponse:
    """Provider-agnostic response for a conversation turn."""

    output_messages: List[ChatMessage]
    usage: Dict[str, Any] | None = None
    raw_response: Any | None = None

    def first_text(self) -> Optional[str]:
        for message in self.output_messages:
            for block in message.content:
                if block.type == "text" and block.text:
                    return block.text
        return None

    def first_reasoning(self) -> Optional[str]:
        for message in self.output_messages:
            for block in message.content:
                if block.type == "reasoning" and block.text:
                    return block.text
        return None


@dataclass
class InferenceConfig:
    """Provider-agnostic inference configuration."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: Sequence[str] | None = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def for_bedrock(self) -> Dict[str, Any]:
        payload = dict(self.extra_params)
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["maxTokens"] = self.max_tokens
        if self.top_p is not None:
            payload["topP"] = self.top_p
        if self.top_k is not None:
            payload["topK"] = self.top_k
        if self.stop_sequences is not None:
            payload["stopSequences"] = list(self.stop_sequences)
        return payload

    def for_gemini(self) -> Dict[str, Any]:
        payload = dict(self.extra_params)
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_output_tokens"] = self.max_tokens
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.stop_sequences is not None:
            payload["stop_sequences"] = list(self.stop_sequences)
        return payload


class LLMClient(ABC):
    """Interface for LLM providers."""

    @abstractmethod
    def converse(
        self,
        *,
        model_id: str,
        messages: Sequence[ChatMessage | Dict[str, Any]],
        inference_config: InferenceConfig | None = None,
        **kwargs: Any,
    ) -> ConverseResponse:
        """Run a conversation turn and return a normalized response."""


class BedrockLLMClient(LLMClient):
    """AWS Bedrock implementation of LLMClient."""

    def __init__(
        self,
        *,
        boto_client: Any | None = None,
        region_name: str | None = None,
        config: Any | None = None,
        **client_kwargs: Any,
    ) -> None:
        if boto_client is not None:
            self._client = boto_client
        else:  # pragma: no cover - boto3 is optional during tests.
            import boto3

            resolved_config = config
            if resolved_config is None and Config is not None:
                resolved_config = Config(read_timeout=900, connect_timeout=900)
            self._client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name or os.getenv("BEDROCK_REGION", "us-east-1"),
                config=resolved_config,
                **client_kwargs,
            )

    def converse(
        self,
        *,
        model_id: str,
        messages: Sequence[ChatMessage | Dict[str, Any]],
        inference_config: InferenceConfig | None = None,
        **kwargs: Any,
    ) -> ConverseResponse:
        normalized_messages = [_ensure_chat_message(msg) for msg in messages]
        serialized_messages = [
            {
                "role": message.role,
                "content": [block.to_bedrock_dict() for block in message.ensure_content()],
            }
            for message in normalized_messages
        ]
        serialized_config = (
            inference_config.for_bedrock() if inference_config is not None else None
        )
        response = self._client.converse(
            modelId=model_id,
            messages=serialized_messages,
            inferenceConfig=serialized_config,
            **kwargs,
        )
        return _bedrock_response_to_converse_response(response)


class GeminiLLMClient(LLMClient):
    """Google Gemini implementation of LLMClient."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str | None = None,
        generation_config: Dict[str, Any] | None = None,
        http_timeout: int | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GeminiLLMClient requires a GEMINI_API_KEY")

        self._api_key = resolved_key
        self._default_model = default_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._endpoint = os.getenv(
            "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        self._http_timeout = http_timeout or 600

        env_include_thoughts = _env_flag("GEMINI_INCLUDE_THOUGHTS")
        if env_include_thoughts is None:
            env_include_thoughts = True  # Thinking summaries are needed downstream.
        env_budget = os.getenv("GEMINI_THINKING_BUDGET")
        env_level = os.getenv("GEMINI_THINKING_LEVEL")
        resolved_generation_config = dict(generation_config or {})
        configured_thinking = resolved_generation_config.pop("thinking_config", None)
        if env_include_thoughts is not None:
            configured_thinking = configured_thinking or {}
            configured_thinking["include_thoughts"] = env_include_thoughts
        if env_budget:
            try:
                configured_thinking = configured_thinking or {}
                configured_thinking["thinking_budget"] = int(env_budget)
            except ValueError:
                pass
        if env_level:
            configured_thinking = configured_thinking or {}
            configured_thinking["thinking_level"] = env_level
        self._default_generation_config = resolved_generation_config
        self._default_thinking_config = configured_thinking

    def converse(
        self,
        *,
        model_id: str,
        messages: Sequence[ChatMessage | Dict[str, Any]],
        inference_config: InferenceConfig | None = None,
        **kwargs: Any,
    ) -> ConverseResponse:
        normalized_messages = [_ensure_chat_message(msg) for msg in messages]
        contents: List[Dict[str, Any]] = [
            {
                "role": message.role,
                "parts": [
                    {"text": block.text} for block in message.ensure_content() if block.text
                ],
            }
            for message in normalized_messages
        ]
        generation_config = dict(self._default_generation_config)
        thinking_config = (
            dict(self._default_thinking_config) if self._default_thinking_config else None
        )
        if inference_config is not None:
            inf_config = inference_config.for_gemini()
            inf_thinking = inf_config.pop("thinking_config", None)
            generation_config.update(inf_config)
            if inf_thinking:
                thinking_config = _merge_optional_dicts(thinking_config, inf_thinking)
        if thinking_config:
            generation_config["thinking_config"] = thinking_config
        payload = {"contents": contents}
        gen_config_payload = _serialize_generation_config(generation_config)
        if gen_config_payload:
            payload["generationConfig"] = gen_config_payload
        model_name = model_id or self._default_model
        endpoint_name = model_name if model_name.startswith("models/") else f"models/{model_name}"
        response = self._post_json(f"{endpoint_name}:generateContent", payload)
        return _gemini_response_to_converse_response(response)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._endpoint}/{path}?key={self._api_key}"
        data = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib_request.urlopen(request, timeout=self._http_timeout) as response:
                body = response.read()
                return json.loads(body.decode("utf-8"))
        except urllib_error.HTTPError as exc:  # pragma: no cover - requires live API call.
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Gemini API request failed ({exc.code}): {detail or exc.reason}"
            ) from exc
        except urllib_error.URLError as exc:  # pragma: no cover - requires live API call.
            raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc


def create_llm_client(provider: str | None = None, **kwargs: Any) -> LLMClient:
    resolved_provider = (provider or os.getenv("LLM_PROVIDER") or "bedrock").lower()
    if resolved_provider == "gemini":
        return GeminiLLMClient(**kwargs)
    if resolved_provider == "bedrock":
        return BedrockLLMClient(**kwargs)
    raise ValueError(f"Unsupported LLM provider: {resolved_provider}")


def _ensure_chat_message(message: ChatMessage | Dict[str, Any]) -> ChatMessage:
    if isinstance(message, ChatMessage):
        return message
    role = message.get("role", "user")
    raw_content = message.get("content", [])
    content_blocks = [_content_block_from_dict(block) for block in raw_content]
    return ChatMessage(role=role, content=content_blocks)


def _content_block_from_dict(data: Dict[str, Any]) -> ContentBlock:
    if "text" in data:
        return ContentBlock(text=data["text"], type="text")
    if "reasoningContent" in data:
        reasoning = data["reasoningContent"]
        reasoning_text = reasoning.get("reasoningText") or reasoning.get("text") or ""
        return ContentBlock(text=reasoning_text, type="reasoning", metadata=reasoning)
    metadata = dict(data)
    text = metadata.get("text", "")
    content_type = metadata.get("type", "unknown")
    return ContentBlock(text=text, type=content_type, metadata=metadata)


def _bedrock_response_to_converse_response(response: Dict[str, Any]) -> ConverseResponse:
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = [
        _content_block_from_dict(block) for block in message.get("content", [])
    ]
    chat_message = ChatMessage(role=message.get("role", "assistant"), content=content_blocks)
    usage = response.get("usage")
    return ConverseResponse(output_messages=[chat_message], usage=usage, raw_response=response)


def _gemini_response_to_converse_response(response: Any) -> ConverseResponse:
    candidates = _get_attr(response, "candidates") or []
    content_blocks: List[ContentBlock] = []
    if candidates:
        primary = candidates[0]
        content = _get_attr(primary, "content")
        parts = _get_attr(content, "parts") if content else None
        if parts is None and isinstance(primary, dict):
            parts = primary.get("content", {}).get("parts")
        if parts:
            for part in parts:
                text = _get_attr(part, "text")
                if not text:
                    continue
                is_thought = bool(_get_attr(part, "thought", False))
                metadata = {"thought": True} if is_thought else None
                block_type = "reasoning" if is_thought else "text"
                content_blocks.append(
                    ContentBlock(text=text, type=block_type, metadata=metadata)
                )
    if not content_blocks:
        text = _get_attr(response, "text", "") or ""
        if text:
            content_blocks.append(ContentBlock(text=text, type="text"))
    usage_metadata = _get_attr(response, "usage_metadata") or _get_attr(response, "usageMetadata")
    usage = None
    if usage_metadata is not None:
        prompt_tokens = _get_attr(usage_metadata, "prompt_token_count")
        output_tokens = _get_attr(usage_metadata, "candidates_token_count")
        total_tokens = _get_attr(usage_metadata, "total_token_count")
        thinking_tokens = _get_attr(usage_metadata, "thoughts_token_count")
        if isinstance(usage_metadata, dict):
            prompt_tokens = usage_metadata.get("promptTokenCount", prompt_tokens)
            output_tokens = usage_metadata.get("candidatesTokenCount", output_tokens)
            total_tokens = usage_metadata.get("totalTokenCount", total_tokens)
            thinking_tokens = usage_metadata.get("thoughtsTokenCount", thinking_tokens)
        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "thinking_tokens": thinking_tokens,
        }
    chat_message = ChatMessage(role="model", content=content_blocks)
    return ConverseResponse(output_messages=[chat_message], usage=usage, raw_response=response)


def _merge_optional_dicts(
    base: Dict[str, Any] | None, override: Dict[str, Any] | None
) -> Dict[str, Any]:
    if base is None and override is None:
        return {}
    if base is None:
        return dict(override or {})
    if override is None:
        return dict(base)
    merged = dict(base)
    merged.update(override)
    return merged


def _env_flag(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _serialize_generation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    thinking_config = config.pop("thinking_config", None)
    for key, value in config.items():
        if value is None:
            continue
        camel_key = _GENERATION_CONFIG_KEY_MAP.get(key, key)
        payload[camel_key] = value
    if thinking_config:
        payload["thinkingConfig"] = _serialize_thinking_config(thinking_config)
    return payload


def _serialize_thinking_config(config: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in config.items():
        if value is None:
            continue
        camel_key = _THINKING_CONFIG_KEY_MAP.get(key, key)
        payload[camel_key] = value
    return payload


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


_GENERATION_CONFIG_KEY_MAP = {
    "max_tokens": "maxOutputTokens",
    "max_output_tokens": "maxOutputTokens",
    "top_p": "topP",
    "top_k": "topK",
    "stop_sequences": "stopSequences",
    "candidate_count": "candidateCount",
}

_THINKING_CONFIG_KEY_MAP = {
    "include_thoughts": "includeThoughts",
    "thinking_budget": "thinkingBudget",
    "thinking_level": "thinkingLevel",
}
