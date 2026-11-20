"""Shared configuration for both FastAPI and Gradio apps."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


def _load_yaml_defaults() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "model_config.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


YAML_DEFAULTS = _load_yaml_defaults()
MODEL_DEFAULTS = YAML_DEFAULTS.get("model", {})
GENERATION_DEFAULTS = YAML_DEFAULTS.get("generation", {})
API_DEFAULTS = YAML_DEFAULTS.get("api", {})
FEATURE_DEFAULTS = YAML_DEFAULTS.get("features", {})


class Settings(BaseModel):
    model_name: str = os.getenv("MODEL_NAME", MODEL_DEFAULTS.get("name", "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"))
    model_dtype: str = os.getenv("MODEL_DTYPE", MODEL_DEFAULTS.get("dtype", "float16"))
    max_model_len: int = int(os.getenv("MAX_MODEL_LEN", str(MODEL_DEFAULTS.get("max_model_len", 8192))))
    gpu_memory_utilization: float = float(
        os.getenv("GPU_MEMORY_UTILIZATION", str(MODEL_DEFAULTS.get("gpu_memory_utilization", 0.9)))
    )

    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

    fastapi_host: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
    fastapi_port: int = int(os.getenv("FASTAPI_PORT", "8100"))

    demo_username: str = os.getenv("FASTAPI_DEMO_USERNAME", "demo")
    demo_password: str = os.getenv("FASTAPI_DEMO_PASSWORD", "demo123")
    demo_token: str = os.getenv("FASTAPI_DEMO_TOKEN", "demo-secret-token")
    default_system_prompt: str = os.getenv(
        "FASTAPI_DEFAULT_SYSTEM_PROMPT",
        API_DEFAULTS.get("default_system_prompt", "You are a concise coding assistant."),
    )

    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", str(GENERATION_DEFAULTS.get("temperature", 0.2))))
    default_top_p: float = float(os.getenv("DEFAULT_TOP_P", str(GENERATION_DEFAULTS.get("top_p", 0.9))))
    default_max_tokens: int = int(os.getenv("DEFAULT_MAX_TOKENS", str(GENERATION_DEFAULTS.get("max_tokens", 512))))

    gradio_host: str = os.getenv("GRADIO_HOST", "0.0.0.0")
    gradio_port: int = int(os.getenv("GRADIO_PORT", "7860"))
    fastapi_base_url: str = os.getenv("FASTAPI_BASE_URL", "http://localhost:8100")

    allow_custom_system_prompt: bool = _get_bool(
        "ALLOW_CUSTOM_SYSTEM_PROMPT",
        FEATURE_DEFAULTS.get("allow_custom_system_prompt", True),
    )
    enable_summary_endpoint: bool = _get_bool(
        "ENABLE_SUMMARY_ENDPOINT",
        FEATURE_DEFAULTS.get("enable_summary_endpoint", True),
    )


settings = Settings()
