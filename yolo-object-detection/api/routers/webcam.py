from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.services.streams.webcam_service import webcam_stream

router = APIRouter()


@router.get("/webcam")
def detect_webcam():
    return StreamingResponse(
        webcam_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

  # =====================
    # http://127.0.0.1:8000/detect/webcam
    # =====================