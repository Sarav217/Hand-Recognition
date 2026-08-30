# Real-Time Hand Finger Counter

This is a Python computer vision project that uses your webcam to detect hand gestures and count the number of fingers you are holding up in real time. It uses **OpenCV** for camera feed rendering and **MediaPipe Hands** for state-of-the-art 3D hand landmark detection.

## Features
- **Real-Time Detection:** Processes your webcam feed seamlessly.
- **Two-Hand Recognition:** Detects both hands and counts total fingers from 0 to 10.
- **Gesture Recognition:** Identifies Thumbs Up, Pointing, Victory/Peace, Open Hand, Both Hands Open, and Fist.
- **Temporal Smoothing:** Reduces jitter by analyzing the past few frames, offering a smooth experience.
- **Modern UI Overlay:** Clean aesthetics with dynamic colors for different finger counts, transparent badges, and intuitive visual feedback.

## Installation

1. Make sure you have **Python 3.10–3.12** installed. The legacy MediaPipe Hands API is not compatible with Python 3.14 or MediaPipe 1.x.
2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the dependencies using the same Python interpreter you will use to run the app:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Usage

### Browser website

Start the Flask server:
```bash
python app.py
```
Then open **http://localhost:8000** in your browser. The server uses the computer's webcam and streams the processed video to the page. Press `Ctrl+C` in the terminal to stop it.

Run the main script to start the application:
```bash
python main.py
```

If you see `ModuleNotFoundError: No module named 'cv2'`, run:
```bash
python -m pip install opencv-python
```
The pip package is named `opencv-python`, while the Python import is `cv2`. On Windows,
use `py -m pip ...` and `py main.py` if `python` is not available.

- Show your hand to the webcam.
- Hold up 0 to 5 fingers.
- The UI will dynamically update to reflect the number of fingers detected.
- Press **'Q'** or close the window to exit the application.

## Gestures Explained

- **0:** Closed fist.
- **1-4:** Hold up the corresponding number of fingers. (e.g., Index for 1; Index + Middle for 2, etc.)
- **5:** Open hand with all fingers and thumb extended.
