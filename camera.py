import cv2

class Camera:
    def __init__(self, index: int = 0):
        # cv2.CAP_DSHOW improves webcam stability on Windows
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("[Camera] Warning: could not open camera.")

    def get_frame(self):
        if not self.cap.isOpened():
            return None
        success, frame = self.cap.read()
        if not success:
            return None
        return frame

    def __del__(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
