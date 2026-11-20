"""Shared request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChatRequest(BaseModel):
    user_message: str = Field(min_length=1)
    temperature: float = 0.2
    max_tokens: int = 512


class ChatResponse(BaseModel):
    reply: str
    model: str


class SummaryRequest(BaseModel):
    content: str = Field(min_length=1)
    max_tokens: int = 256


class PromptChatRequest(BaseModel):
    system_prompt: str = Field(min_length=1)
    user_message: str = Field(min_length=1)
    temperature: float = 0.2
    max_tokens: int = 512


class GenericResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    model: str
    vllm_base_url: str
