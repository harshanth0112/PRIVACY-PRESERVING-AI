# backend/detector.py  (PyTorch .pt using Ultralytics YOLO) - UPDATED with IoU tracker & persistence
import os
import time
import json
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("ultralytics package required. Install in venv: pip install ultralytics") from e

BASE_DIR = os.path.dirname(__file__)
MODEL_PT = os.path.join(BASE_DIR, "models", "best.pt")   # ensure this path
NAMES_PATH = os.path.join(BASE_DIR, "models", "names.txt")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LOG_PATH = os.path.join(BASE_DIR, "logs", "events.log")
ALERTS_JSON = os.path.join(BASE_DIR, "logs", "alerts.json")


def ensure_logs():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(ALERTS_JSON):
        with open(ALERTS_JSON, "w", encoding="utf-8") as f:
            json.dump([], f)


class Detector:
    def __init__(self, model_path: str = None):
        ensure_logs()

        # load config or create defaults
        default_cfg = {
            "alarm_classes": ["knife", "gun", "fire"],
            "confidence_threshold": 0.65,
            "iou_threshold": 0.65,
            "cooldown_seconds": 5,
            "frame_skip": 0,
            "input_size": 640,
            "alarm_window_seconds": 3,
            "alarm_window_min_hits": 1
        }
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = default_cfg
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(default_cfg, f, indent=4)
        for k, v in default_cfg.items():
            cfg.setdefault(k, v)
        self.config = cfg

        # model path
        mp = model_path if model_path else MODEL_PT
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Model file not found: {mp}")

        # load YOLOv8 .pt
        print("[Detector] Loading YOLO model from:", mp)
        self.model = YOLO(mp)  # uses ultralytics -> PyTorch under the hood

        # try to read names from model; fallback to names.txt
        try:
            self.class_names = self.model.names
        except Exception:
            if os.path.exists(NAMES_PATH):
                with open(NAMES_PATH, "r", encoding="utf-8") as f:
                    self.class_names = [ln.strip() for ln in f.readlines() if ln.strip()]
            else:
                self.class_names = {}

        self.latest: Dict[str, Any] = {"timestamp": None, "detections": [], "alarm_active": False, "alarm_reason": ""}
        self.last_alarm_time = 0.0
        self.alarm_history: List[float] = []
        self._frame_counter = 0
        self.frame_skip = int(self.config.get("frame_skip", 0))
        self.input_size = int(self.config.get("input_size", 640))
        self.last_annotated = None
        self.last_detections: List[Dict[str, Any]] = []

        # -----------------------
        # Simple IoU-based tracker (persistence)
        # -----------------------
        self.tracks: List[Dict[str, Any]] = []  # each track: {id, bbox, cls, last_seen, hits, score_sum}
        self.next_track_id = 1
        self.track_iou_thresh = 0.35            # IoU threshold to match detections to tracks
        self.track_max_age = 1.5                # seconds to keep track alive without being seen
        self.track_required_hits = 3            # require N hits to consider persistent (tuneable)

        print("[Detector] Model loaded. Classes:", len(self.class_names) if hasattr(self.class_names, '__len__') else "unknown")
        print(f"[Detector] Tracker: iou_thresh={self.track_iou_thresh}, max_age={self.track_max_age}, required_hits={self.track_required_hits}")

    def _log_event(self, message: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}] {message}\n"
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
        print(line, end="")

    def _record_alert_json(self, alarm_reason: str):
        try:
            with open(ALERTS_JSON, "r+", encoding="utf-8") as f:
                arr = json.load(f)
                arr.append({"ts": time.time(), "reason": alarm_reason})
                f.seek(0)
                json.dump(arr[-100:], f, indent=2)
        except Exception:
            with open(ALERTS_JSON, "w", encoding="utf-8") as f:
                json.dump([{"ts": time.time(), "reason": alarm_reason}], f, indent=2)

    def take_local_action(self, alarm_reason: str):
        try:
            self._record_alert_json(alarm_reason)
            if os.name == "nt":
                import winsound
                winsound.Beep(1000, 300)
        except Exception as e:
            print("[take_local_action] failed:", e)

    # -----------------------
    # Utility: IoU
    # -----------------------
    @staticmethod
    def _iou(a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        xi1 = max(ax1, bx1)
        yi1 = max(ay1, by1)
        xi2 = min(ax2, bx2)
        yi2 = min(ay2, by2)
        w = max(0, xi2 - xi1)
        h = max(0, yi2 - yi1)
        inter = w * h
        areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = areaA + areaB - inter + 1e-6
        return inter / union

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Run the PyTorch .pt model (Ultralytics) on a frame and return annotated frame + detections list.
        Detections: {"class":str, "conf":float, "bbox":[x1,y1,x2,y2]}
        """
        self._frame_counter += 1
        do_infer = (self._frame_counter % (self.frame_skip + 1) == 0)
        if not do_infer and self.last_annotated is not None:
            return self.last_annotated, self.last_detections

        # Run model (ultralytics yields Results list)
        results = self.model(frame, imgsz=self.input_size, verbose=False)  # returns list
        res = results[0]

        detections: List[Dict[str, Any]] = []
        conf_thresh = float(self.config.get("confidence_threshold", 0.45))
        alarm_classes = set(self.config.get("alarm_classes", []))

        # res.boxes: each box -> .xyxy, .conf, .cls
        if hasattr(res, "boxes") and res.boxes is not None:
            boxes = res.boxes
            for b in boxes:
                # YOLOv8 boxes: xyxy tensor
                try:
                    xyxy = b.xyxy[0].cpu().numpy()
                    conf = float(b.conf[0]) if hasattr(b, "conf") else float(b.conf)
                    cls_id = int(b.cls[0]) if hasattr(b, "cls") else int(b.cls)
                except Exception:
                    # fallback to older API
                    xyxy = b.xyxy.cpu().numpy().reshape(-1).tolist()
                    conf = float(b.conf.cpu().numpy().reshape(-1)[0])
                    cls_id = int(b.cls.cpu().numpy().reshape(-1)[0])

                x1, y1, x2, y2 = map(int, xyxy[:4])
                class_name = self.class_names.get(cls_id, str(cls_id)) if isinstance(self.class_names, dict) else (self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id))
                if conf < conf_thresh:
                    # optionally save weak frames for later labeling:
                    # if class_name in alarm_classes:
                    #     os.makedirs(os.path.join(BASE_DIR, "hard_samples"), exist_ok=True)
                    #     cv2.imwrite(os.path.join(BASE_DIR, "hard_samples", f"weak_{class_name}_{int(time.time()*1000)}.jpg"), frame)
                    continue
                det = {"class": class_name, "conf": round(conf, 3), "bbox": [x1, y1, x2, y2]}
                detections.append(det)

                # draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # -----------------------
        # Tracker update (match detections to existing tracks)
        # -----------------------
        now = time.time()
        # Match detections -> tracks by class + IoU
        for det in detections:
            bx = det["bbox"]
            cls = det["class"]
            conf = det["conf"]
            matched = False
            # look for best matching track (same class)
            best_idx = None
            best_iou = 0.0
            for i, tr in enumerate(self.tracks):
                if tr["cls"] != cls:
                    continue
                iou_val = self._iou(tr["bbox"], bx)
                if iou_val >= self.track_iou_thresh and iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i
            if best_idx is not None:
                # update existing track
                tr = self.tracks[best_idx]
                tr["bbox"] = bx
                tr["last_seen"] = now
                tr["hits"] += 1
                tr["score_sum"] += conf
            else:
                # new track
                new_tr = {
                    "id": self.next_track_id,
                    "bbox": bx,
                    "cls": cls,
                    "last_seen": now,
                    "hits": 1,
                    "score_sum": conf
                }
                self.tracks.append(new_tr)
                self.next_track_id += 1

        # prune old tracks
        active_tracks = []
        for tr in self.tracks:
            if (now - tr["last_seen"]) <= self.track_max_age:
                active_tracks.append(tr)
        self.tracks = active_tracks

        # -----------------------
        # Alarm decision based on persistent tracks (required hits)
        # -----------------------
        alarm_triggered = False
        alarm_reason = ""
        # find the best persistent alarm track (highest avg conf)
        best_track = None
        best_avg = 0.0
        for tr in self.tracks:
            if tr["cls"] in alarm_classes and tr["hits"] >= self.track_required_hits:
                avg_conf = tr["score_sum"] / max(1, tr["hits"])
                if avg_conf > best_avg:
                    best_avg = avg_conf
                    best_track = tr

        cooldown = float(self.config.get("cooldown_seconds", 5))
        if best_track is not None and (now - self.last_alarm_time) > cooldown:
            alarm_triggered = True
            alarm_reason = f"{best_track['cls']} persistent (avg_conf={best_avg:.2f})"
            self.last_alarm_time = now
            self._log_event(alarm_reason)
            self.take_local_action(alarm_reason)
            # reset hits for that track to avoid immediate re-trigger
            best_track["hits"] = 0
            best_track["score_sum"] = 0.0

        alarm_active = alarm_triggered

        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.latest = {"timestamp": ts, "detections": detections, "alarm_active": alarm_active, "alarm_reason": alarm_reason}
        self.last_detections = detections
        self.last_annotated = frame.copy()
        return frame, detections

    def update_latest(self, detections: List[Dict[str, Any]]):
        """
        Keep API compatible with app.py and add clustering info.
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # image size fallback
        img_w, img_h = None, None
        try:
            if self.last_annotated is not None:
                h0, w0 = self.last_annotated.shape[:2]
                img_w, img_h = w0, h0
        except Exception:
            img_w, img_h = self.input_size, self.input_size

        # centers
        centers = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        radius = max(40, int(max(img_w, img_h) * 0.12))
        clusters = []
        for idx, (cx, cy) in enumerate(centers):
            placed = False
            for cl in clusters:
                ccx, ccy = cl["centroid"]
                dist = ((cx - ccx) ** 2 + (cy - ccy) ** 2) ** 0.5
                if dist <= radius:
                    cl["members"].append(idx)
                    n = len(cl["members"])
                    cl["centroid"] = ((ccx * (n - 1) + cx) / n, (ccy * (n - 1) + cy) / n)
                    bx1, by1, bx2, by2 = cl["bbox"]
                    dx1, dy1, dx2, dy2 = detections[idx]["bbox"]
                    cl["bbox"] = [min(bx1, dx1), min(by1, dy1), max(bx2, dx2), max(by2, dy2)]
                    cls = detections[idx]["class"]
                    cl["class_counts"][cls] = cl["class_counts"].get(cls, 0) + 1
                    placed = True
                    break
            if not placed:
                x1, y1, x2, y2 = detections[idx]["bbox"]
                cls = detections[idx]["class"]
                clusters.append({"members": [idx], "bbox": [x1, y1, x2, y2], "centroid": (cx, cy), "class_counts": {cls: 1}})

        clusters_out = []
        for cl in clusters:
            clusters_out.append({
                "members": cl["members"],
                "bbox": [int(cl["bbox"][0]), int(cl["bbox"][1]), int(cl["bbox"][2]), int(cl["bbox"][3])],
                "total": len(cl["members"]),
                "class_counts": cl["class_counts"]
            })

        alarm_active = getattr(self, "latest", {}).get("alarm_active", False)
        alarm_reason = getattr(self, "latest", {}).get("alarm_reason", "")

        self.latest = {
            "timestamp": ts,
            "detections": detections,
            "clusters": clusters_out,
            "alarm_active": alarm_active,
            "alarm_reason": alarm_reason
        }

    def get_latest(self) -> Dict[str, Any]:
        return self.latest
