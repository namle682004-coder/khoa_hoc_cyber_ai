"""vLLM and gateway HTTP client helpers."""
from __future__ import annotations

import httpx

from .config import settings


async def vllm_chat_completion(messages: list[dict], temperature: float, max_tokens: int) -> str:
    payload = {
        "model": settings.model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(f"{settings.vllm_base_url}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"]


def gateway_login(base_url: str, username: str, password: str) -> str:
    payload = {"username": username, "password": password}
    response = httpx.post(f"{base_url}/auth/login", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["access_token"]


def gateway_post(base_url: str, path: str, token: str, payload: dict) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    response = httpx.post(f"{base_url}{path}", json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    return response.json()["reply"]
