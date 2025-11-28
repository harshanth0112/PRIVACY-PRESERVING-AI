from flask import Flask, Response, jsonify, send_from_directory
from camera import Camera
from detector import Detector
import os

app = Flask(__name__, static_folder="../frontend", static_url_path="")

# create detector + camera
camera = Camera()
detector = Detector()  # uses config.json or defaults

def generate_frames():
    """
    Grab frame from camera, run detector, update latest, and stream JPEG frames.
    """
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        annotated_frame, detections = detector.process_frame(frame)
        detector.update_latest(detections)

        # JPEG encode
        import cv2
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        jpg_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')

@app.route("/")
def index():
    # serve frontend index.html
    return send_from_directory(os.path.join(app.root_path, '../frontend'), 'index.html')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detections")
def detections():
    return jsonify(detector.get_latest())

if __name__ == "__main__":
    # Run on 0.0.0.0 so other LAN devices can view (optional)
    app.run(host="0.0.0.0", port=8000, debug=False)
