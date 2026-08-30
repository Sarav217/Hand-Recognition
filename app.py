"""Flask web interface for the real-time hand finger counter."""
import atexit
import threading
import time

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit("OpenCV is missing. Run: python -m pip install -r requirements.txt") from exc

from flask import Flask, Response, render_template
from hand_detector import HandDetector
from ui_overlay import UIOverlay

app = Flask(__name__)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
detector = HandDetector(max_hands=2, detection_con=0.7, track_con=0.7, smooth_frames=7)
ui = UIOverlay()
camera_lock = threading.Lock()
state_lock = threading.Lock()
latest_state = {"left": None, "right": None, "total": None, "mode": False, "hands": {}}


def frames():
    while True:
        with camera_lock:
            success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        frame = detector.find_hands(frame, draw=True)
        result = detector.snapshot()
        total, gesture = result["total"], result["gesture"]
        with state_lock:
            latest_state.update({**result, "mode": total == 10, "hands": detector.hand_points.copy(), "timestamp": time.time()})
        label = "-" if total is None else f"{result['left'] if result['left'] is not None else '-'}{result['right'] if result['right'] is not None else '-'}"
        frame = ui.draw_hud(frame, total, gesture, label)
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if success:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/state")
def state():
    with state_lock:
        return latest_state.copy()


@atexit.register
def cleanup():
    camera.release()
    detector.close()


if __name__ == "__main__":
    if not camera.isOpened():
        raise SystemExit("Could not open webcam. Check camera permissions or camera index 0.")
    print("Open http://localhost:8000 in your browser")
    app.run(host="localhost", port=8000, threaded=True, debug=False)
