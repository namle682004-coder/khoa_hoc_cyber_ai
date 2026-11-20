import os
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO


# ======================================
# Download data from Roboflow
# ======================================

def download_dataset(dataset_name, api_key=None, workspace=None, version=1, target_dir="./data"):
    """
    Download data từ Roboflow Universe

    Args:
        dataset_name (str)
        api_key (str)
        workspace (str)
        version (int)
        target_dir (str)

    Returns:
        dataset_path (str)
    """

    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("Install roboflow: pip install roboflow")

    if api_key is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")

    if api_key is None:
        raise ValueError("Missing Roboflow API key")

    rf = Roboflow(api_key=api_key)

    project = rf.workspace(workspace).project(dataset_name)
    dataset = project.version(version).download(
        model_format="yolov8",
        location=target_dir,
    )

    return dataset.location


# ======================================
# Plot Detection Results
# ======================================

def plot_detection_results(results, class_names=None):
    """
    Visualize YOLO detection results

    Args:
        results: ultralytics YOLO results
        class_names: optional list
    """

    r = results[0]

    img = r.orig_img.copy()

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        label = (
            f"{class_names[cls_id]} {conf:.2f}"
            if class_names
            else f"{cls_id} {conf:.2f}"
        )

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


# ======================================
# Calculate mAP
# ======================================

def calculate_map(model: YOLO, data_yaml):
    """
    Calculate mAP using YOLO validation

    Args:
        model: ultralytics YOLO model
        data_yaml: path to data.yaml

    Returns:
        metrics dict
    """

    metrics = model.val(data=data_yaml)

    return {
        "map50": metrics.box.map50,
        "map50_95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }


# ======================================
# Count objects by class
# ======================================

def count_objects_by_class(results):
    """
    Count number of detected objects per class

    Args:
        results: YOLO inference results

    Returns:
        dict {class_name: count}
    """

    counter = Counter()

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            counter[cls_name] += 1

    return dict(counter)
