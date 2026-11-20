from fastapi import APIRouter, UploadFile, File, Query

from api.services.detectors.image_service import detect_image
from api.schemas.detection import ImageResponse

router = APIRouter()


@router.post("/image", response_model=ImageResponse)
async def detect_image_api(
    file: UploadFile = File(...),
    conf: float = Query(0.5, ge=0.05, le=1.0),
):

    content = await file.read()

    path, detections = detect_image(content, conf)

    return {
        "image_path": path,
        "detections": detections,
    }
