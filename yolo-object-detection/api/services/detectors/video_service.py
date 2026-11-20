import uuid
import cv2

from api.core.model_loader import model
from api.core.config import Config


def detect_video(video_path: str, conf: float):

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Config.VIDEO_OUTPUT / f"{uuid.uuid4().hex}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=conf,
            device=Config.DEVICE,
            verbose=False,
        )

        annotated = results[0].plot()
        writer.write(annotated)

    cap.release()
    writer.release()

    return out_path
