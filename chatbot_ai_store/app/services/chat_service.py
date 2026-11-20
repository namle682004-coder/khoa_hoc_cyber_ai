from fastapi import FastAPI
from fastapi.exceptions import ValidationException

from api.core.settings import settings
from api.middlewares.auth_jwt import JWTMiddleware
from api.routers import image, video, webcam
from api.core.logger import setup_logging
from api.middlewares.error_handler import unhandled_exception_handler
from api.middlewares.validation_handler import validation_exception_handler
from api.middlewares.metrics_middleware import MetricsMiddleware
from api.middlewares.rate_limit import RateLimitMiddleware
from api.middlewares.request_id import RequestIDMiddleware
from api.routers.metrics import router as metrics_router
from api.infra.redis import init_redis, close_redis
from contextlib import asynccontextmanager

setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_redis()
    yield
    await close_redis()


app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
)
@app.get("/")
def root():
    return {"status": "ok", "service": settings.APP_NAME}

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "gpu": settings.DEVICE,
        "redis": "connected"
    }

app.add_middleware(JWTMiddleware)
app.add_exception_handler(Exception, unhandled_exception_handler)
app.add_exception_handler(ValidationException, validation_exception_handler)
app.add_middleware(MetricsMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestIDMiddleware)

app.include_router(metrics_router)
app.include_router(image.router, prefix="/detect", tags=["Image"])
app.include_router(video.router, prefix="/detect", tags=["Video"])
app.include_router(webcam.router, prefix="/detect", tags=["Webcam"])
