from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    APP_NAME: str = "YOLO Detection API"

    DEBUG: bool = True
    # Path model
    MODEL_PATH: str = "checkpoints/best.pt"

    # Device
    DEVICE: str = "cuda"

    # API key
    API_KEY: str = "dev-secret-key"

    # JWT
    JWT_SECRET: str = "super-secret"
    JWT_ALGORITHM: str = "HS256"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_DB: int = 0
    # GPU
    GPU_INDEX: int = 0

    # Rate limit
    RATE_LIMIT_PER_MIN: int = 60

    class Config:
        env_file = ".env"


settings = Settings()
