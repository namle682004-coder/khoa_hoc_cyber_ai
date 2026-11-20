import uuid
import cv2
import numpy as np

from api.core.model_loader import model
from config import Config


def detect_image(image_bytes: bytes, conf: float):

    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model.predict(
        img,
        conf=conf,
        device=Config.DEVICE,
        verbose=False,
    )

    annotated = results[0].plot()

    path = f"runs/images/{uuid.uuid4().hex}.jpg"
    cv2.imwrite(path, annotated)

    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "class_name": model.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        })

    return path, detections
