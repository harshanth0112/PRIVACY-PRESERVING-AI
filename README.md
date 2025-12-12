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
🛠️ Installation Guide
1. Create Virtual Environment
Open your terminal (CMD) inside the project folder and create a Python 3.10 environment:

Bash

# Verify you have Python 3.10 installed
py -3.10 -m venv venv310

# Activate the environment
venv310\Scripts\activate
(You should see (venv310) at the start of your command line).

2. Install Dependencies
Navigate to the backend folder and install the strict versions required:

Bash

cd backend
pip install ultralytics==8.2.0
pip install opencv-python
pip install flask
pip install numpy==1.26.4
⚠️ Critical: You must use numpy==1.26.4. Newer versions (2.x) will break PyTorch on Windows.

3. Setup Model
Place your trained YOLOv8 model file (best.pt) inside the backend/models/ directory. The model should be trained on classes like: fire, smoke, knife, gun, pistol.

▶️ Usage
Start the Backend: From the backend folder (with venv activated), run:

Bash

python app.py
The server will start at http://127.0.0.1:8000.

Launch the Frontend: Open the frontend/index.html file directly in your web browser.

Test the System:

Show a knife or gun (real or on a phone screen).

Play a video of fire/smoke.

The system will beep and display a red bounding box if the threat persists.

Stop: Press CTRL + C in the terminal to stop the server.

⚙️ Configuration
You can tune detection sensitivity in backend/config.json:

JSON

{
  "confidence_threshold": 0.28,
  "iou_threshold": 0.45,
  "cooldown_seconds": 4,
  "alarm_window_seconds": 2,
  "alarm_window_min_hits": 1
}
⚠️ Disclaimer
This project is for educational and research purposes. Please adhere to all local laws regarding surveillance and privacy when deploying this system.
