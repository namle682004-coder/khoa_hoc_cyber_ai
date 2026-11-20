from ultralytics import YOLO
from api.core.config import Config

print("Loading YOLO model:", Config.CHECKPOINT_PATH)

model = YOLO(str(Config.CHECKPOINT_PATH))
