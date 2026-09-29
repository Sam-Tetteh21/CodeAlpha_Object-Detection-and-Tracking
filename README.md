# CodeAlpha_ObjectDetectionTracking

A web app that detects and tracks objects in an uploaded video, built during my **CodeAlpha Artificial Intelligence Internship** (Task 4).

Upload a video, and it draws a labeled box around every detected object (person, car, dog, etc. — 80 categories total), with a persistent ID number that follows each object across frames.

## 🌍 Live Demo
👉 [Try it live here](https://codealphaobject-detection-and-tracking.streamlit.app)

## 🎥 Video Demo
[LinkedIn video link here]

## ✨ Features
- Upload any MP4/AVI/MOV video
- Detects 80 everyday object categories using a pre-trained YOLOv4-tiny model
- Tracks each object with a persistent ID number as it moves across frames
- Adjustable frame-processing rate to trade off speed vs. smoothness
- Download the finished, annotated video

## 🛠️ Tech Stack & How It Works
- **Python 3**
- **[OpenCV](https://opencv.org/)** — loads and runs the pre-trained YOLOv4-tiny model (via its DNN module), reads/writes video
- **YOLOv4-tiny** — a neural network already trained (by its creators, on the COCO dataset) to recognize 80 object classes. No training happens in this project — the model is used exactly as downloaded.
- **A custom centroid tracker** — a lightweight tracking algorithm (in the same family as SORT) that gives each detected object a persistent ID by matching object positions between consecutive frames
- **[Streamlit](https://streamlit.io/)** — the web interface

**In plain terms:** every video frame is fed into YOLOv4-tiny, which returns a list of detected objects with bounding boxes and labels. Since the model has no memory between frames, a separate tracker compares each frame's detections to the previous frame's tracked objects (by how close their centers are) to decide "is this the same object as before, or a new one?" — that's what gives each object a stable ID instead of a new random label every frame.

## 📂 Project Structure
```
CodeAlpha_ObjectDetectionTracking/
├── app.py               # Streamlit UI — upload, process, display, download
├── detector.py           # YOLOv4-tiny loading + running detection on a frame
├── tracker.py             # Centroid-based multi-object tracker
├── model/
│   ├── yolov4-tiny.weights   # Pre-trained model weights
│   ├── yolov4-tiny.cfg        # Model architecture config
│   └── coco.names              # The 80 object class names the model recognizes
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 How to Run Locally

1. Clone this repository
   ```bash
   git clone https://github.com/<your-username>/CodeAlpha_ObjectDetectionTracking.git
   cd CodeAlpha_ObjectDetectionTracking
   ```

2. (Optional) create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app
   ```bash
   streamlit run app.py
   ```

5. Upload a short video, click **Run Detection & Tracking**, and wait for it to process (a progress bar shows frame-by-frame status). Longer/higher-resolution videos take longer since every frame runs through the neural network.

## 📌 About the Internship
This project was built as part of the **CodeAlpha Artificial Intelligence Internship**.

- Website: [www.codealpha.tech](https://www.codealpha.tech)
- Task: Object Detection and Tracking

## 👤 Author
**Samuel Tetteh**
BSc. Information Technology Education, Level 300
University of Skills Training and Entrepreneurial Development (USTED)

🔗 [LinkedIn](www.linkedin.com/in/samuel-tetteh-b5a247356) · [GitHub](https://github.com/Sam-Tetteh21)
