import redis.asyncio as redis
from api.core.settings import settings

redis_client: redis.Redis | None = None


async def init_redis():
    global redis_client

    redis_client = redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
    )


async def close_redis():
    global redis_client

    if redis_client:
        await redis_client.close()
