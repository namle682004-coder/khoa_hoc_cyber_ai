"""FastAPI gateway application."""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .auth import verify_bearer_token
from .config import settings
from .schemas import (
    ChatRequest,
    ChatResponse,
    GenericResponse,
    HealthResponse,
    LoginRequest,
    LoginResponse,
    PromptChatRequest,
    SummaryRequest,
)
from .vllm_client import vllm_chat_completion

app = FastAPI(
    title="Chapter4 Unified Gateway",
    version="1.0.0",
    description="One-codebase gateway for vLLM, FastAPI, and Gradio demo.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=GenericResponse)
async def root() -> GenericResponse:
    return GenericResponse(message="Chapter4 API is running")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model=settings.model_name, vllm_base_url=settings.vllm_base_url)


@app.post("/auth/login", response_model=LoginResponse)
async def login(payload: LoginRequest) -> LoginResponse:
    if payload.username != settings.demo_username or payload.password != settings.demo_password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return LoginResponse(access_token=settings.demo_token)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _: bool = Depends(verify_bearer_token)) -> ChatResponse:
    messages = [
        {"role": "system", "content": settings.default_system_prompt},
        {"role": "user", "content": req.user_message},
    ]
    reply = await vllm_chat_completion(messages, req.temperature, req.max_tokens)
    return ChatResponse(reply=reply, model=settings.model_name)


@app.post("/api/summary", response_model=ChatResponse)
async def summary(req: SummaryRequest, _: bool = Depends(verify_bearer_token)) -> ChatResponse:
    if not settings.enable_summary_endpoint:
        raise HTTPException(status_code=404, detail="Summary endpoint is disabled")
    messages = [
        {"role": "system", "content": "Summarize user content into concise bullet points in Vietnamese."},
        {"role": "user", "content": req.content},
    ]
    reply = await vllm_chat_completion(messages, 0.1, req.max_tokens)
    return ChatResponse(reply=reply, model=settings.model_name)


@app.post("/api/chat-with-system-prompt", response_model=ChatResponse)
async def chat_with_system_prompt(req: PromptChatRequest, _: bool = Depends(verify_bearer_token)) -> ChatResponse:
    if not settings.allow_custom_system_prompt:
        raise HTTPException(status_code=403, detail="Custom system prompt is disabled")
    messages = [
        {"role": "system", "content": req.system_prompt},
        {"role": "user", "content": req.user_message},
    ]
    reply = await vllm_chat_completion(messages, req.temperature, req.max_tokens)
    return ChatResponse(reply=reply, model=settings.model_name)


@app.on_event("startup")
async def startup_log() -> None:
    logger.info("FastAPI started | model={} | vllm={}", settings.model_name, settings.vllm_base_url)
