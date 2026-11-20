from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
from api.infra.redis import redis_client

RATE_LIMIT = 60
WINDOW = 60


class RateLimitMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        if redis_client is None:
            return await call_next(request)

        ip = request.client.host
        key = f"rl:{ip}"

        count = await redis_client.incr(key)

        if count == 1:
            await redis_client.expire(key, WINDOW)

        if count > RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"},
            )

        return await call_next(request)
