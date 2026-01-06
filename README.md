# 🚗 AI-Powered ADAS: Hazard Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green)

### 📄 Project Overview
This project is an **Advanced Driver Assistance System (ADAS)** prototype designed to enhance road safety. It uses **Computer Vision** and **Deep Learning** to detect vehicles, pedestrians, and traffic signs in real-time video feeds.

The system processes dashcam footage frame-by-frame, identifies potential hazards, and issues **collision warnings** based on proximity and object type.

### 🌟 Key Features
* **Real-Time Object Detection:** Identifies Cars, Trucks, Buses, and Pedestrians.
* **Safety Alerts:** Triggers visual warnings (e.g., "⚠️ CAUTION: PEDESTRIAN") when high-risk objects are detected.
* **Traffic Analytics:** Counts the number of vehicles on screen to estimate traffic density.
* **Interactive Dashboard:** Built with Streamlit for easy video upload and visualization.

### 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit
* **AI Model:** YOLOv8 (Medium/Nano) by Ultralytics
* **Computer Vision:** OpenCV

### 🚀 How to Run Locally
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/ADAS-Project.git](https://github.com/YOUR_USERNAME/ADAS-Project.git)
    cd ADAS-Project
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

### 🧠 How It Works
1.  **Input:** The user uploads a dashcam video file (MP4).
2.  **Processing:** OpenCV breaks the video into individual frames (images).
3.  **Inference:** The **YOLOv8 CNN model** scans each frame to detect objects and assign a confidence score (e.g., "Car: 0.85").
4.  **Logic Layer:** The Python script analyzes the detections. If a person is found or traffic is dense, it overlays a warning message.
5.  **Output:** The annotated video is streamed back to the user in real-time.

---
*Created by Soham Kumar Paul | B.Tech CSE (AI-ML)*
