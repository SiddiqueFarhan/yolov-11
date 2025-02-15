import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Change to "yolov11" when available

# Streamlit UI
st.title("Live Object Detection with YOLO")
st.write("Turn on the webcam and detect objects in real-time.")

# Start/Stop button
run = st.checkbox("Start Webcam")

# Open webcam stream if checkbox is checked
if run:
    # OpenCV Video Capture
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    # Create a placeholder for video output
    frame_placeholder = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.error("Failed to access webcam.")
            break

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO inference
        results = model(frame_rgb)

        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Display the frame in Streamlit
        frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)

        # Stop if user unchecks the box
        if not st.session_state["checkbox"]:
            break

    cap.release()
    cv2.destroyAllWindows()
