from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse


async def validation_exception_handler(request: RequestValidationError , exc:Exception) -> JSONResponse:
    return JSONResponse(
        status_code=400,
             content={"detail": exc.errors(),
                      },
      )
