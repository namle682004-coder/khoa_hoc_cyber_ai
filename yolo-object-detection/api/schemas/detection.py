from pydantic import BaseModel
from typing import List
from api.core.logger import get_logger

logger = get_logger(__name__)

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: BBox

class ImageResponse(BaseModel):
    image_path: str
    detections: List[Detection]
