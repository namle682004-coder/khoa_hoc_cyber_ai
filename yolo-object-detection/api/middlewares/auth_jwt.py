from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import jwt

from api.core.settings import settings


PUBLIC_PREFIXES = (
    "/",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/metrics",
)


class JWTMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        path = request.url.path

        # ✅ Allow public endpoints
        if path.startswith(PUBLIC_PREFIXES):
            return await call_next(request)

        auth = request.headers.get("Authorization")

        if not auth:
            raise HTTPException(status_code=401, detail="Missing token")

        parts = auth.split()

        if len(parts) != 2:
            raise HTTPException(status_code=401, detail="Invalid Authorization header")

        scheme, token = parts

        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid auth scheme")

        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM],
            )

            request.state.user = payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")

        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

        return await call_next(request)
