"""
CodeAlpha - Object Detection and Tracking
Task 4: AI Internship

Upload a video file, and this app runs a pre-trained YOLOv4-tiny model
on every frame to detect objects, then tracks each object across
frames with a persistent ID number, drawing labeled boxes on the
output video.
"""

import os
import tempfile

import cv2
import streamlit as st

from detector import load_model, detect_objects
from tracker import CentroidTracker

st.set_page_config(page_title="CodeAlpha Object Detection & Tracking", page_icon="🎥")
st.title("🎥 Object Detection and Tracking")
st.caption("CodeAlpha AI Internship - Task 4")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov4-tiny.weights")
CONFIG_PATH = os.path.join(BASE_DIR, "yolov4-tiny.cfg")
NAMES_PATH = os.path.join(BASE_DIR, "coco.names")


@st.cache_resource
def get_model():
    return load_model(WEIGHTS_PATH, CONFIG_PATH, NAMES_PATH)


net, output_layers, classes = get_model()

st.markdown(
    "Upload a short video (MP4/AVI/MOV). The app detects **80 common object "
    "types** (people, cars, animals, everyday items, etc.) using a pre-trained "
    "YOLOv4-tiny model, and tracks each one with a persistent ID as it moves."
)

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

frame_skip = st.slider(
    "Process every Nth frame (higher = faster, choppier tracking)",
    min_value=1,
    max_value=5,
    value=1,
)

if uploaded_file is not None:
    # Save the uploaded video to a temp file so OpenCV can open it by path
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_temp.write(uploaded_file.read())
    input_temp.close()

    st.video(input_temp.name)

    if st.button("Run Detection & Tracking", type="primary"):
        cap = cv2.VideoCapture(input_temp.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        tracker = CentroidTracker(max_disappeared=30, max_distance=80)

        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_index = 0
        last_detections = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only run the (relatively slow) YOLO forward pass every
            # Nth frame; reuse the last known boxes on skipped frames
            # so the output video stays smooth.
            if frame_index % frame_skip == 0:
                last_detections = detect_objects(net, output_layers, classes, frame)

            boxes = [d["box"] for d in last_detections]
            labels = [d["label"] for d in last_detections]

            tracked_objects = tracker.update(boxes)

            # Draw each tracked box with its label + persistent ID
            for object_id, (x, y, w, h) in tracked_objects.items():
                # Find the label for this box (matched by position)
                label = "object"
                for box, lbl in zip(boxes, labels):
                    if box == (x, y, w, h):
                        label = lbl
                        break

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"ID {object_id}: {label}"
                cv2.putText(
                    frame, text, (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            writer.write(frame)

            frame_index += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_index / total_frames, 1.0))
            status_text.text(f"Processing frame {frame_index}/{total_frames or '?'}...")

        cap.release()
        writer.release()

        status_text.text("Done!")
        progress_bar.progress(1.0)

        st.subheader("Result")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                "Download annotated video",
                data=f,
                file_name="detected_output.mp4",
                mime="video/mp4",
            )

st.divider()
st.caption("Built with Python, OpenCV (YOLOv4-tiny), and a centroid-based tracker.")
