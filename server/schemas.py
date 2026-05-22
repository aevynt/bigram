"""Pydantic schemas for the Bigram Tensor 1 API."""

from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    recurrence: int | None = None
    temperature: float = 0.7
    top_p: float | None = 0.9
    top_k: int | None = 40
    repetition_penalty: float = 1.1
    abstention_threshold: float | None = None


class GenerateResponse(BaseModel):
    text: str
    abstained: bool = False
    error: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class AgentChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tool_turns: int = 6
    recurrence: int | None = 64
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float | None = 0.9
    top_k: int | None = 40
    repetition_penalty: float = 1.1


class AgentChatResponse(BaseModel):
    answer: str
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    abstained: bool = False
    error: str | None = None


class RagSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class RagSearchResponse(BaseModel):
    results: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
