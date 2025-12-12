
# 🔒 Edge YOLO Surveillance – Privacy-Preserving AI

A **real-time, edge-based surveillance system** powered by **YOLOv8** for detecting weapons (knives, guns, pistols) and hazards (fire, smoke).  
All video processing happens **locally on your machine** — no cloud uploads, ensuring **maximum privacy**.

---

## 🚀 Features

- **Real-Time Detection**: Fire, smoke, knives, guns, pistols via webcam/laptop camera  
- **Privacy-First**: 100% local inference, no internet required  
- **Smart Alarm System**:  
  - Persistence check (≥2 frames) prevents false alarms  
  - Audible alerts (Windows system beep)  
  - Event logging to `backend/logs/events.log`  
- **Live Dashboard**: Browser-based UI with bounding boxes and system status  

---

## 📋 System Requirements

| Component   | Requirement                  |
|-------------|------------------------------|
| **OS**      | Windows 10 / 11              |
| **Python**  | 3.10.x (strictly required)   |
| **RAM**     | 4 GB minimum (8 GB recommended) |
| **Camera**  | USB webcam or integrated laptop camera |

⚠️ Python 3.11+ is **not supported** due to YOLOv8 + PyTorch compatibility issues  
⚠️ `numpy==1.26.4` is mandatory (newer 2.x versions break PyTorch on Windows)

---

## 📂 Project Structure

```text
EdgeYOLO_Project/
├── backend/
│   ├── models/
│   │   └── best.pt        # Trained YOLO model
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
```

---

## 🛠️ Installation Guide

### 1. Create Python 3.10 Virtual Environment
```bash
py -3.10 -m venv venv310
venv310\Scripts\activate
```

### 2. Install Dependencies
```bash
cd backend
pip install ultralytics==8.2.0
pip install opencv-python
pip install flask
pip install numpy==1.26.4
```

### 3. Add Your YOLO Model
Place your trained model in:
```bash
backend/models/best.pt
```
Supported classes: `fire`, `smoke`, `knife`, `gun`, `pistol`

---

## ▶️ Running the System

1. **Start Backend**
```bash
python app.py
```
Backend runs at: `http://127.0.0.1:8000`

2. **Launch Frontend**  
Open `frontend/index.html` in any browser

3. **Test Detection**  
- Show a knife/gun (real or on phone)  
- Play fire/smoke video  

System will:  
✔ Draw bounding boxes  
✔ Beep  
✔ Log event  

4. **Stop Server**  
Press `CTRL + C` in terminal

---

## ⚙️ Configuration (`backend/config.json`)

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

This project is for **educational and research purposes only**.  
Ensure compliance with **local surveillance and privacy laws** before real-world deployment.

---

## ⭐ Contributions

Pull requests, feature requests, and enhancements are welcome!  
Help improve privacy-preserving AI for edge devices.
