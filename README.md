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
## 🛠️ Installation Guide

### 1. Create a Python 3.10 Virtual Environment

```bash
# Ensure Python 3.10 is installed
py -3.10 -m venv venv310

# Activate the environment
venv310\Scripts\activate
2. Install Dependencies
Navigate to the backend directory:

bash
Copy code
cd backend
Install required packages:

bash
Copy code
pip install ultralytics==8.2.0
pip install opencv-python
pip install flask
pip install numpy==1.26.4
⚠️ Important:
numpy==1.26.4 is mandatory. Newer 2.x versions break PyTorch on Windows.

3. Add Your YOLO Model
Place your trained model file:

bash
Copy code
backend/models/best.pt
The model must be trained on one or more classes:
fire, smoke, knife, gun, pistol

▶️ Running the System
1. Start the Backend
bash
Copy code
python app.py
Backend runs at:

cpp
Copy code
http://127.0.0.1:8000
2. Launch the Frontend
Open:

bash
Copy code
frontend/index.html
directly in any browser.

3. Test Detection
Show a knife/gun (real or on phone).

Play a fire/smoke video.

The system will:
✔ draw bounding boxes
✔ beep
✔ log the event

4. Stop the Server
Press CTRL + C in the terminal.

⚙️ Configuration (backend/config.json)
Adjust detection sensitivity:

json
Copy code
{
  "confidence_threshold": 0.28,
  "iou_threshold": 0.45,
  "cooldown_seconds": 4,
  "alarm_window_seconds": 2,
  "alarm_window_min_hits": 1
}
⚠️ Disclaimer
This project is intended for educational and research use only.
Before deploying in real-world environments, ensure compliance with all applicable surveillance and privacy laws.

⭐ Contributions
Pull requests and enhancements are welcome.
Feel free to submit issues or feature requests.
