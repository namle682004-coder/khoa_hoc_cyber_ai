import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from api.monitoring.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from api.monitoring.gpu import gpu_stats


class MetricsMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        start = time.time()

        response = await call_next(request)

        elapsed = time.time() - start

        REQUEST_LATENCY.labels(request.url.path).observe(elapsed)

        REQUEST_COUNT.labels(
            request.method,
            request.url.path,
            response.status_code,
        ).inc()

        stats = gpu_stats()

        return response
