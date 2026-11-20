"""Gradio UI application sharing same codebase as FastAPI."""
from __future__ import annotations

import gradio as gr

from .config import settings
from .vllm_client import gateway_login, gateway_post

TOKEN_STATE = {"token": ""}


def do_login() -> str:
    try:
        token = gateway_login(settings.fastapi_base_url, settings.demo_username, settings.demo_password)
        TOKEN_STATE["token"] = token
        return "Dang nhap thanh cong"
    except Exception as exc:  # noqa: BLE001
        return f"Dang nhap that bai: {exc}"


def run_chat(user_message: str, temperature: float, max_tokens: int) -> str:
    if not TOKEN_STATE["token"]:
        return "Vui long bam Login truoc."
    payload = {"user_message": user_message, "temperature": temperature, "max_tokens": max_tokens}
    try:
        return gateway_post(settings.fastapi_base_url, "/api/chat", TOKEN_STATE["token"], payload)
    except Exception as exc:  # noqa: BLE001
        return f"Loi chat: {exc}"


def run_summary(content: str, max_tokens: int) -> str:
    if not TOKEN_STATE["token"]:
        return "Vui long bam Login truoc."
    payload = {"content": content, "max_tokens": max_tokens}
    try:
        return gateway_post(settings.fastapi_base_url, "/api/summary", TOKEN_STATE["token"], payload)
    except Exception as exc:  # noqa: BLE001
        return f"Loi summary: {exc}"


def run_custom_chat(system_prompt: str, user_message: str, temperature: float, max_tokens: int) -> str:
    if not TOKEN_STATE["token"]:
        return "Vui long bam Login truoc."
    payload = {
        "system_prompt": system_prompt,
        "user_message": user_message,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        return gateway_post(settings.fastapi_base_url, "/api/chat-with-system-prompt", TOKEN_STATE["token"], payload)
    except Exception as exc:  # noqa: BLE001
        return f"Loi custom prompt chat: {exc}"


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="Chapter4 Unified Demo") as demo:
        gr.Markdown("# Chapter 4 Demo: Gradio -> FastAPI -> vLLM")

        login_btn = gr.Button("Login with demo account", variant="primary")
        login_output = gr.Textbox(label="Auth status", interactive=False)
        login_btn.click(fn=do_login, outputs=login_output)

        with gr.Tab("Chat"):
            chat_input = gr.Textbox(label="User message", lines=4)
            chat_temp = gr.Slider(0.0, 1.5, value=settings.default_temperature, step=0.1, label="Temperature")
            chat_tokens = gr.Slider(64, 1024, value=settings.default_max_tokens, step=32, label="Max tokens")
            chat_btn = gr.Button("Send")
            chat_out = gr.Textbox(label="Assistant reply", lines=14)
            chat_btn.click(run_chat, [chat_input, chat_temp, chat_tokens], chat_out)

        with gr.Tab("Summary"):
            summary_input = gr.Textbox(label="Content", lines=8)
            summary_tokens = gr.Slider(64, 1024, value=min(256, settings.default_max_tokens), step=32, label="Max tokens")
            summary_btn = gr.Button("Summarize")
            summary_out = gr.Textbox(label="Summary", lines=12)
            summary_btn.click(run_summary, [summary_input, summary_tokens], summary_out)

        with gr.Tab("Custom System Prompt"):
            system_prompt = gr.Textbox(label="System prompt", lines=5, value=settings.default_system_prompt)
            custom_input = gr.Textbox(label="User message", lines=4)
            custom_temp = gr.Slider(0.0, 1.5, value=settings.default_temperature, step=0.1, label="Temperature")
            custom_tokens = gr.Slider(64, 1024, value=settings.default_max_tokens, step=32, label="Max tokens")
            custom_btn = gr.Button("Run with custom system prompt")
            custom_out = gr.Textbox(label="Assistant reply", lines=14)
            custom_btn.click(run_custom_chat, [system_prompt, custom_input, custom_temp, custom_tokens], custom_out)
    return demo


def main() -> None:
    app = create_ui()
    app.launch(server_name=settings.gradio_host, server_port=settings.gradio_port)


if __name__ == "__main__":
    main()
