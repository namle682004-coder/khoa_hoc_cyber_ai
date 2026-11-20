FROM vllm/vllm-openai:latest

WORKDIR /app

COPY config/model_config.yaml /app/config/model_config.yaml

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]