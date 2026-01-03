import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

# --- PAGE SETUP ---
st.set_page_config(page_title="ADAS AI Dashboard", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1 {
        color: #FF4B4B;
        text-align: center;
    }
    .stAlert {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Intelligent ADAS System")
st.markdown("### Real-time Hazard Detection & Traffic Analysis")
st.write("This system uses **Computer Vision (YOLOv8)** to detect vehicles and pedestrians, estimating safety risks.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4)
st.sidebar.info("Higher confidence means fewer but more accurate detections.")

uploaded_file = st.sidebar.file_uploader("Upload Dashcam Video (.mp4)", type=['mp4'])

# --- AI MODEL LOADING ---
# This downloads the model automatically on first run
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- MAIN PROCESSING ---
if uploaded_file is not None:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    
    col1, col2, col3 = st.columns(3)
    kpi1 = col1.empty()
    kpi2 = col2.empty()
    kpi3 = col3.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. AI Detection
        results = model(frame, conf=confidence)
        
        # 2. Extract Data
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Count objects in this frame
        boxes = results[0].boxes
        car_count = 0
        truck_count = 0
        person_count = 0
        
        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            if class_name == 'car': car_count += 1
            if class_name == 'truck': truck_count += 1
            if class_name == 'person': person_count += 1

        # 3. Add Safety Warnings (The ADAS part)
        # If a person is found, flash a warning on the video
        if person_count > 0:
            cv2.putText(annotated_frame, "⚠️ CAUTION: PEDESTRIAN", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        elif car_count > 5:
            cv2.putText(annotated_frame, "TRAFFIC: HEAVY", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
        else:
            cv2.putText(annotated_frame, "STATUS: CLEAR", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # 4. Display Dashboard Stats
        kpi1.metric("Cars Detected", car_count)
        kpi2.metric("Trucks/Buses", truck_count)
        
        if person_count > 0:
             kpi3.error(f"Pedestrians: {person_count}")
        else:
             kpi3.success("Pedestrians: 0")

        # 5. Show Video
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
else:
    st.info("👈 Waiting for video upload...")
