import gradio as gr
import requests
import tempfile
import cv2
from PIL import Image

# ==============================
# CONFIG
# ==============================

API_BASE = "http://127.0.0.1:8000/detect"

REQUEST_TIMEOUT_IMAGE = 120
REQUEST_TIMEOUT_VIDEO = 900

# cache frames sau detect video
VIDEO_FRAMES = []


# ==============================
# IMAGE API
# ==============================

def run_image(image: Image.Image, conf: float):

    if image is None:
        return None, []

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(tmp.name)

    with open(tmp.name, "rb") as f:
        resp = requests.post(
            f"{API_BASE}/image",
            params={"conf": conf},
            files={"file": f},
            timeout=REQUEST_TIMEOUT_IMAGE,
        )

    resp.raise_for_status()
    data = resp.json()

    img = cv2.imread(data["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    table = []

    for d in data["detections"]:

        bbox = d["bbox"]

        if isinstance(bbox, dict):
            x1, y1, x2, y2 = bbox.values()

        elif isinstance(bbox, list):
            x1, y1, x2, y2 = bbox

        else:
            x1 = y1 = x2 = y2 = "?"

        table.append([
            d["class_name"],
            round(d["confidence"], 3),
            f"[{x1}, {y1}, {x2}, {y2}]",
        ])

    return img, table


# ==============================
# VIDEO API
# ==============================

def run_video(video_path: str, conf: float):

    global VIDEO_FRAMES
    VIDEO_FRAMES = []

    if video_path is None:
        return None, gr.update(visible=False), None

    # call backend
    with open(video_path, "rb") as f:
        resp = requests.post(
            f"{API_BASE}/video",
            params={"conf": conf},
            files={"file": f},
            timeout=REQUEST_TIMEOUT_VIDEO,
        )

    resp.raise_for_status()

    # save detected video
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out.write(resp.content)
    out.close()

    # decode frames ONCE
    cap = cv2.VideoCapture(out.name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        VIDEO_FRAMES.append(frame)

    cap.release()

    if not VIDEO_FRAMES:
        return out.name, gr.update(visible=False), None

    return (
        out.name,
        gr.update(
            maximum=len(VIDEO_FRAMES) - 1,
            value=0,
            visible=True,
        ),
        VIDEO_FRAMES[0],
    )


# ==============================
# SEEK FRAME (RAM CACHE)
# ==============================

def seek_frame(frame_idx: int):

    if not VIDEO_FRAMES:
        return None

    frame_idx = int(frame_idx)

    if frame_idx < 0 or frame_idx >= len(VIDEO_FRAMES):
        return None

    return VIDEO_FRAMES[frame_idx]


# ==============================
# UI
# ==============================

def build_ui():

    with gr.Blocks(
        title="YOLOv8 Professional Detection Suite",
    ) as demo:

        gr.Markdown(
            """
            # 🚀 YOLOv8 Professional Detection Suite
            Detect Image • Video (Seek Frame) • Webcam realtime từ FastAPI
            """
        )

        with gr.Tabs():

            # ================= IMAGE ================= #

            with gr.Tab("🖼 Image Detection"):

                with gr.Row():
                    input_image = gr.Image(type="pil", label="Upload Image")
                    output_image = gr.Image(label="Detected Image")

                conf_slider = gr.Slider(0.1, 1.0, 0.5, step=0.05, label="Confidence Threshold")
                detect_btn = gr.Button("▶ Run Detection", variant="primary")

                detection_table = gr.Dataframe(
                    headers=["Class", "Confidence", "BBox"],
                    interactive=False,
                )

                detect_btn.click(
                    fn=run_image,
                    inputs=[input_image, conf_slider],
                    outputs=[output_image, detection_table],
                )

            # ================= VIDEO ================= #

            with gr.Tab("🎥 Video Detection + Seek"):

                input_video = gr.Video(label="Upload Video")

                video_conf = gr.Slider(
                    0.1, 1.0, 0.5, step=0.05,
                    label="Confidence Threshold"
                )

                detect_video_btn = gr.Button("▶ Run Video Detection", variant="primary")

                output_video = gr.Video(label="Detected Video")

                frame_slider = gr.Slider(
                    0, 100, value=0, step=1,
                    visible=False,
                    label="Seek Frame",
                )

                frame_preview = gr.Image(label="Frame Preview")

                detect_video_btn.click(
                    fn=run_video,
                    inputs=[input_video, video_conf],
                    outputs=[output_video, frame_slider, frame_preview],
                )

                frame_slider.change(
                    fn=seek_frame,
                    inputs=frame_slider,
                    outputs=frame_preview,
                )

            # ================= WEBCAM ================= #

            # ================= WEBCAM ================= #

            with gr.Tab("📷 Webcam Live"):
                gr.Markdown("Realtime webcam detection từ FastAPI backend")

                webcam_html = gr.HTML(
                    "<div style='font-size:14px;color:#888'>Webcam is stopped</div>"
                )

                with gr.Row():
                    start_btn = gr.Button("▶ Start Webcam", variant="primary")
                    stop_btn = gr.Button("⏹ Stop Webcam")

                def start_webcam():
                    return """
                    <img src="http://127.0.0.1:8000/detect/webcam"
                         width="720"
                         style="border-radius:12px;border:1px solid #ddd">
                    """

                def stop_webcam():
                    return "<div style='font-size:14px;color:#888'>Webcam stopped</div>"

                start_btn.click(
                    fn=start_webcam,
                    outputs=webcam_html,
                )

                stop_btn.click(
                    fn=stop_webcam,
                    outputs=webcam_html,
                )

    return demo


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    app = build_ui()

    app.launch(
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),  # đúng chuẩn gradio >= 6
    )
