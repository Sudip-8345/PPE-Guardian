# PPE Guardian – Real-Time Safety Violation Detection Using Computer Vision

## Overview

PPE Guardian is a real-time computer vision system that automatically detects Personal Protective Equipment (PPE) compliance in industrial environments. It identifies workers and checks whether mandatory safety gear (helmet, vest, gloves, boots, mask) is being used. If any required PPE is missing, the system flags a violation and logs the event with bounding box evidence and tracking ID.

This project aims to reduce manual safety monitoring effort and increase workplace compliance in industries such as:
- Construction
- Manufacturing plants
- Warehouses and logistics hubs
- Mining and refinery operations

---

## Key Features

- Real-time PPE detection using YOLOv8
- Detects: Helmet, Vest, Gloves, Boots, Mask and Person
- Worker tracking with ByteTrack to maintain ID across frames
- Violation detection logic (Example: Person without Helmet)
- Annotated video output with bounding boxes and labels
- Violation log stored in CSV with timestamp, bounding box, and track ID
- Modern Streamlit dashboard for UI
- Supports webcam and video upload
- Supports both PyTorch and ONNX inference modes

---

## Tech Stack

| Component | Technology Used |
|----------|------------------|
| Model Training | YOLOv8 (Ultralytics) |
| Tracking | ByteTrack |
| User Interface | Streamlit |
| Backend | Python |
| Inference | PyTorch, ONNX Runtime |
| Data Annotation Format | YOLOv8 |
| Experiments | Google Colab |

---

## Dataset

Source: Roboflow Universe  
Project: PPE Detection – PRAM  
Dataset: 3000+ industrial images  
Classes:

- Hardhat (Helmet)
- Person
- Safety Boots
- Safety Gloves
- Safety Mask
- Safety Vest

Dataset Download Script (Colab):

```python
from roboflow import Roboflow 
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("pram").project("ppe-detection-z3v2w")
version = project.version(2)
dataset = version.download("yolov8")

## Application Demo

### Real-Time Detection
![PPE Guardian Demo](https://github.com/Sudip-8345/PPE-Guardian/blob/e4f871328c0631fef4e83dacbdcc542082ebccd2/assests/demo.png)

### Training Metrics (YOLOv8 Learning Curves)
![Training Metrics](https://github.com/Sudip-8345/PPE-Guardian/blob/c278855af74b3be04562e9c5de49802eba45ba40/assests/results.png)

### Dataset Class Distribution & Label Visualization
![Dataset Stats](https://github.com/Sudip-8345/PPE-Guardian/blob/c278855af74b3be04562e9c5de49802eba45ba40/assests/labels.jpg)

### Confusion Matrix
![Confusion Matrix](https://github.com/Sudip-8345/PPE-Guardian/blob/c278855af74b3be04562e9c5de49802eba45ba40/assests/confusion_matrix.png)

