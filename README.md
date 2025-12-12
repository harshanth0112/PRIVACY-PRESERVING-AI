# Edge YOLO Surveillance - Privacy-Preserving AI

A real-time, privacy-preserving surveillance system that uses a local YOLOv8 model to detect weapons (knives, guns) and fire. The system processes video feeds entirely on the edge (your local machine), ensuring no video data is sent to the cloud.

## 🚀 Features

* **Real-Time Detection**: Detects fire, smoke, knives, guns, and pistols using a webcam or laptop camera.
* **Privacy-First**: All inference happens locally; no internet connection required.
* **Smart Alarm System**:
    * **Persistence Check**: Triggers only if an object persists for 2+ frames to prevent false alarms.
    * **Audible Alerts**: Plays a system beep (Windows) upon confirmed detection.
    * **Logging**: Events are recorded in `backend/logs/events.log`.
* **Live Dashboard**: Browser-based UI showing the live feed with bounding boxes and status.

## 📋 System Requirements

* **OS**: Windows 10/11.
* **Python**: **3.10.x** (Strictly required; 3.12/3.13 are not supported).
* **RAM**: 4GB minimum (8GB recommended).
* **Hardware**: USB Webcam or integrated laptop camera.

## 📂 Project Structure

Ensure your project is organized as follows:

```text
EdgeYOLO_Project/
├── backend/
│   ├── models/
│   │   └── best.pt       # <--- Your trained YOLO model goes here
│   ├── app.py
│   ├── camera.py
│   ├── detector.py
│   └── config.json
├── frontend/
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   └── app.js
│   └── index.html
└── README.md
