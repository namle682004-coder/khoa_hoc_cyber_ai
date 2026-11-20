import argparse
import os
import time

import cv2
from ultralytics import YOLO

from config import Config
from utils import count_objects_by_class


# =====================================
# Inference on Image
# =====================================

def inference_image(checkpoint_path, image_path, conf_threshold=0.5):
    """
    1. Load trained YOLO checkpoint
    2. Run inference trên single image
    3. Visualize bounding boxes với class labels
    4. Print detected objects và confidence scores
    5. Save annotated image
    """

    model = YOLO(checkpoint_path)

    print(f"[INFO] Running inference on image: {image_path}")

    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        device=Config.DEVICE,
        save=False,
        verbose=False,
    )

    r = results[0]

    annotated_img = r.plot()

    save_path = os.path.join(
        os.path.dirname(image_path),
        f"annotated_{os.path.basename(image_path)}",
    )

    cv2.imwrite(save_path, annotated_img)

    print(f"[INFO] Saved annotated image to: {save_path}")

    print("\nDetected objects:")
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]

        print(f"- {cls_name}: {conf:.3f}")

    counts = count_objects_by_class(results)

    print("\nObject counts by class:")
    for k, v in counts.items():
        print(f"{k}: {v}")

    return save_path


# =====================================
# Inference on Video (Bonus)
# =====================================

def inference_video(checkpoint_path, video_path, output_path):
    """
    1. Process video frame by frame
    2. Draw bounding boxes on each frame
    3. Save output video
    4. Print FPS và total detections
    """

    model = YOLO(checkpoint_path)

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_detections = 0
    frame_count = 0
    start_time = time.time()

    print(f"[INFO] Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=0.5,
            device=Config.DEVICE,
            verbose=False,
        )

        annotated = results[0].plot()

        writer.write(annotated)

        frame_count += 1
        total_detections += len(results[0].boxes)

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    fps_proc = frame_count / elapsed if elapsed > 0 else 0

    print(f"[INFO] Saved output video to: {output_path}")
    print(f"[INFO] Processed FPS: {fps_proc:.2f}")
    print(f"[INFO] Total detections: {total_detections}")


# =====================================
# Command Line Interface
# =====================================

def parse_args():
    parser = argparse.ArgumentParser("YOLOv8 Inference")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="output.mp4")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if args.image:
        inference_image(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            conf_threshold=args.conf_threshold,
        )

    elif args.video:
        inference_video(
            checkpoint_path=args.checkpoint,
            video_path=args.video,
            output_path=args.output,
        )

    else:
        print("❌ Please provide either --image or --video")
