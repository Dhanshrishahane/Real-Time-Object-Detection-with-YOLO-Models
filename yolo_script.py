import random
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import base64
import tempfile

# Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: left;
        }}
        .block-container {{
            margin-left: 0;
            margin-right: auto;
        }}
        h1 {{
            font-size: 3em;
            font-weight: bold;
            color: #FFF;
            text-align: center;
        }}
        h2, h3 {{
            font-size: 2em;
            font-weight: bold;
            color: #FFF;
        }}
        .stButton > button {{
            font-size: 1.2em;
        }}
        .stTextInput > div > div > input {{
            font-size: 1.2em;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Set background image
set_background(r"C:\\Users\\dhana\\OneDrive\\Desktop\\NIT_studyMaterial\\All_AI(Projects)\\Object_detection(YOLO)\\dec2.jpg")

# YOLO model paths
models = {
    "YOLOv8n": "weights/yolov8n.pt",
    "YOLOv8s": "weights/yolov8s.pt",
    "YOLOv8m": "weights/yolov8m.pt",
    "YOLOv8l": "weights/yolov8l.pt",
    "YOLOv8x": "weights/yolov8x.pt",
}

# Streamlit app setup
st.title("Real-Time Object Detection with YOLO Models")
st.write("This project uses various YOLO models to detect and classify objects in real-time through a webcam feed or from uploaded images and videos.")

# Button to display descriptions of YOLO models
if st.button("Description of Models"):
    st.markdown("""
    **YOLO Model Descriptions:**
    - **YOLOv8n (Nano)**: The smallest model in YOLOv8, best for devices with limited computational power.
    - **YOLOv8s (Small)**: Slightly larger than Nano, with improved accuracy but still lightweight.
    - **YOLOv8m (Medium)**: A balanced model for accuracy and speed.
    - **YOLOv8l (Large)**: High accuracy, suited for scenarios where speed is less critical.
    - **YOLOv8x (Extra Large)**: The most accurate but computationally intensive YOLOv8 variant.
    """)

# Model selection dropdown
model_choice = st.selectbox("Choose YOLO Model", list(models.keys()))

# Load the selected YOLO model
model = YOLO(models[model_choice])

# Read class names from file
coco_file = r"C:\\Users\\dhana\\OneDrive\\Desktop\\NIT_studyMaterial\\All_AI(Projects)\\Object_detection(YOLO)\\coco.txt"
try:
    with open(coco_file, "r") as my_file:
        class_list = my_file.read().strip().split("\n")
except FileNotFoundError:
    st.error(f"Class names file not found: {coco_file}")
    st.stop()

# Generate random colors for class list
detection_colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in class_list
]

# Webcam and File Input
st.subheader("Upload Image or Video or Use Webcam")
uploaded_file = st.file_uploader("Choose a file (Image/Video)", type=["jpg", "jpeg", "png", "mp4", "avi"])

# Webcam controls
st.subheader("Real-Time Detection from Webcam")
use_webcam = st.checkbox("Use Webcam for Detection")
start_detection = st.button("Start Webcam Detection")
stop_detection = st.button("Stop Webcam Detection")

# Placeholder for video frames
frame_placeholder = st.empty()

# Function to process image
def process_image(image):
    results = model.predict(source=image, conf=0.45, save=False)
    detections = results[0]

    for box in detections.boxes:
        clsID = int(box.cls[0])
        conf = box.conf[0]
        bb = box.xyxy[0]

        # Draw the bounding box
        cv2.rectangle(
            image,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[clsID],
            2,
        )

        # Display class name and confidence
        label = f"{class_list[clsID]}: {conf:.2f}"
        cv2.putText(
            image,
            label,
            (int(bb[0]), int(bb[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    # Display the annotated image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(image, channels="RGB")

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.45, save=False)
        detections = results[0]

        for box in detections.boxes:
            clsID = int(box.cls[0])
            conf = box.conf[0]
            bb = box.xyxy[0]

            # Draw the bounding box
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                2,
            )

            # Display class name and confidence
            label = f"{class_list[clsID]}: {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (int(bb[0]), int(bb[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

    cap.release()

# Function to process webcam feed
def process_webcam():
    cap = cv2.VideoCapture(0)
    while start_detection:
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access webcam. Please check your camera.")
            break

        results = model.predict(source=frame, conf=0.45, save=False)
        detections = results[0]

        for box in detections.boxes:
            clsID = int(box.cls[0])
            conf = box.conf[0]
            bb = box.xyxy[0]

            # Draw the bounding box
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                2,
            )

            # Display class name and confidence
            label = f"{class_list[clsID]}: {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (int(bb[0]), int(bb[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

    cap.release()

# Process the uploaded file or webcam
if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        process_image(image)

    elif uploaded_file.type.startswith("video"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        process_video(tmp_path)

elif use_webcam and start_detection:
    process_webcam()
