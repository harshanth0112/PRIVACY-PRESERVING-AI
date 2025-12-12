
````markdown
# Edge YOLO Surveillance – Privacy-Preserving AI

A real-time, on-device surveillance system powered by **YOLOv8**, engineered to detect weapons (knife, gun, pistol) and fire hazards without sending any video frames to the cloud. The system processes feeds entirely on the edge, ensuring a privacy-first operational environment.

---

## 🚀 Core Capabilities

- **Real-Time Threat Detection**  
  Identifies fire, smoke, knives, guns, and pistols using any standard webcam.

- **Privacy-First Architecture**  
  All inference runs locally; no cloud upload, no external streaming.

- **Smart Alarm Engine**  
  - Detection triggered only when an object persists for **2+ frames**.  
  - **Windows Beep Alert** upon validated detection.  
  - All threat events logged to:  
    ```
    backend/logs/events.log
    ```

- **Live Monitoring Dashboard**  
  A browser-based interface that shows the live feed with bounding boxes and threat status.

---

## 📋 System Requirements

| Component | Requirement |
|----------|-------------|
| OS | Windows 10 / Windows 11 |
| Python | **3.10.x only** (3.12 / 3.13 not supported) |
| RAM | Minimum 4GB (8GB recommended) |
| Hardware | Laptop/USB Webcam |

---

## 📂 Project Structure

````

EdgeYOLO_Project/
├── backend/
│   ├── models/
│   │   └── best.pt              # YOLOv8 trained model
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

````

---

## 🛠️ Installation Guide

### 1. Create a Python 3.10 Virtual Environment

```bash
# Ensure Python 3.10 is installed
py -3.10 -m venv venv310

# Activate the environment
venv310\Scripts\activate
````

---

### 2. Install Dependencies

Navigate to the backend directory:

```bash
cd backend
```

Install required packages:

```bash
pip install ultralytics==8.2.0
pip install opencv-python
pip install flask
pip install numpy==1.26.4
```

⚠️ **Important:**
`numpy==1.26.4` is mandatory. Newer 2.x versions break PyTorch on Windows.

---

### 3. Add Your YOLO Model

Place your trained model file:

```
backend/models/best.pt
```

The model must be trained on one or more classes:
**fire, smoke, knife, gun, pistol**

---

## ▶️ Running the System

### 1. Start the Backend

```bash
python app.py
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

### 2. Launch the Frontend

Open:

```
frontend/index.html
```

directly in any browser.

---

### 3. Test Detection

* Show a **knife/gun** (real or on phone).
* Play a **fire/smoke** video.
* The system will:
  ✔ draw bounding boxes
  ✔ beep
  ✔ log the event

---

### 4. Stop the Server

Press **CTRL + C** in the terminal.

---

## ⚙️ Configuration (backend/config.json)

Adjust detection sensitivity:

```json
{
  "confidence_threshold": 0.28,
  "iou_threshold": 0.45,
  "cooldown_seconds": 4,
  "alarm_window_seconds": 2,
  "alarm_window_min_hits": 1
}
```

---

## ⚠️ Disclaimer

This project is intended for **educational and research use only**.
Before deploying in real-world environments, ensure compliance with all applicable surveillance and privacy laws.

---

## ⭐ Contributions

Pull requests and enhancements are welcome.
Feel free to submit issues or feature requests.

---

## 📄 License

This project is released under the MIT License.

```

---

If you'd like to add **badges, screenshots, diagrams, or a deployment GIF**, I can integrate those into the README as well for higher impact on GitHub, Boss.
```
