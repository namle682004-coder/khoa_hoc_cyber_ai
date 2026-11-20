
# YOLOv8 Object Detection Project

##  I. Giới thiệu

Project này xây dựng một **hệ thống Object Detection hoàn chỉnh** sử dụng **YOLOv8** bao gồm:

* Training script
* Inference script (ảnh & video)
* Gradio web demo app
* Utilities cho download dataset, visualize, đánh giá mAP

Dataset được lấy từ **Roboflow Universe** (ví dụ: Hard Hat Detection / Face Mask / Pothole Detection).

---

## II. Project Structure

```
yolo-object-detection/
│
├── api/
│   ├── main.py                # entrypoint FastAPI
│   │
│   ├── core/
│   │   ├── config.py          # env + settings
│   │   ├── logger.py
│   │   └── model_loader.py
│   │
│   ├── routes/
│   │   ├── detect.py
│   │   └── health.py
│   │
│   ├── schemas/
│   │   └── detect.py
│   │
│   └── services/
│       └── yolo_service.py
│
├── checkpoints/
│   └── best.pt
│
├── runs/
│   ├── images/
│   └── videos/
│
├── docker/
│   ├── Dockerfile
│
├── .env
├── requirements.txt
├── README.md
└── train.py

```

---

## III. Dataset

Sử dụng dataset từ **Roboflow Universe** (YOLOv8 format):

Ví dụ:

* Hard Hat Detection
* Face Mask Detection
* Pothole Detection

Dataset gồm:

* train/images + labels
* valid/images + labels
* test/images + labels
* data.yaml

---

## IV. Installation

### 1️. Clone repo

```bash
git clone https://github.com/yourname/yolo-object-detection.git
cd yolo-object-detection
```

---

### 2️. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

---

### 3️. Install dependencies

```bash
pip install -r requirements.txt
```

---

## V. Download Dataset

### Option 1: Download thủ công từ Roboflow

1. Vào [https://universe.roboflow.com](https://universe.roboflow.com)
2. Chọn dataset → Export → YOLOv8
3. Giải nén vào folder:

```
data/
└── your-dataset/
    ├── train/
    ├── valid/
    ├── test/
    └── data.yaml
```

Sửa trong `config.py`:

```python
DATA_ROOT = "./data/your-data"
DATA_YAML = "./data/your-data/data.yaml"
```

---

### Option 2: Download bằng script

```bash
python train.sh.py --download \
  --roboflow-key YOUR_API_KEY \
  --workspace WORKSPACE \
  --version 1
```

---

##  VI.️ Training

Chạy training:

```bash
python train.sh.py \
  --epochs 50 \
  --batch-size 16 \
  --img-size 640 \
  --model-size n
```

Checkpoint tốt nhất sẽ được lưu tại:

```
checkpoints/best.pt
```

---

## VII. Inference

---

### 1. Inference ảnh

```bash
python inference.py \
  --checkpoint checkpoints/best.pt \
  --image samples/test.jpg \
  --conf-threshold 0.5
```

Kết quả:

```
samples/annotated_test.jpg
```

---

### 2. Inference video

```bash
python inference.py \
  --checkpoint checkpoints/best.pt \
  --video samples/demo.mp4 \
  --output outputs/result.mp4
```

Console sẽ in:

* FPS
* Tổng detections

---

## VIII. Gradio Demo App

Launch web app:

```bash
python gradio_app.py
```

Sau đó mở link public Gradio hiển thị trên terminal.

Tính năng:

* Upload ảnh
* Upload video
* Điều chỉnh confidence threshold
* Hiển thị bounding boxes
* Bảng object detections

---

## IX. Expected Results

Sau khi training ~50 epochs (YOLOv8n):

| Metric         | Value       |
|----------------| ----------- |
| mAP50          | 0.78 – 0.90 |
| mAP50-95       | 0.45 – 0.65 |
| Precision      | ~0.8        |
| Recall         | ~0.75       |
| FPS (RTX 3050) | 40–70       |

⚠️ Kết quả thay đổi theo dataset & GPU.

---

## X. Sample Outputs

Thư mục `samples/`:

```
samples/
├── image_result.jpg
├── video_result.mp4
```

Ví dụ:

* Helmet detection
* Face mask detection
* Pothole detection

---

## XI. Lessons Learned

Trong quá trình làm project:

### 1. Những gì học được

* Cách training YOLOv8 với Ultralytics
* Format dataset YOLO
* Đánh giá mAP
* Xây dựng pipeline inference
* Deploy demo bằng Gradio
* Sử dụng Roboflow API

---

### 2. Challenges gặp phải

* Dataset imbalance → recall thấp
* GPU memory limit → giảm batch size
* Label sai format
* Overfitting khi dataset nhỏ
* Tune augmentation parameters
* Video inference FPS thấp

---

## 3. Future Improvements

* Hyperparameter tuning
* Mixed precision training
* ONNX / TensorRT export
* Tracking (ByteTrack)
* Stream webcam realtime
* Dockerize project

---

## 4. Author

Student project – YOLOv8 Object Detection
Course: Computer Vision / Deep Learning

---

Nếu bạn muốn, mình có thể:

✅ viết `requirements.txt`
✅ chỉnh README theo dataset cụ thể bạn chọn
✅ thêm screenshot mẫu
✅ viết report PDF nộp bài

