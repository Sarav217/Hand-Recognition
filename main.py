"""Entry point for the webcam finger counter."""
try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit("OpenCV is missing. Run: python -m pip install -r requirements.txt\n(The pip package is opencv-python; the import name is cv2.)") from exc
from hand_detector import HandDetector
from ui_overlay import UIOverlay

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open webcam. Check camera permissions or index 0.")

    # Initialize the hand detector with a temporal smoothing buffer
    detector = HandDetector(max_hands=2, detection_con=0.7, track_con=0.7, smooth_frames=7)
    
    # Initialize the modern UI overlay manager
    ui = UIOverlay()
    
    print("Starting Real-Time Hand Finger Counter... Press 'Q' to exit.")

    try:
        while True:
            success, img = cap.read()
            if not success or img is None:
                print("Warning: camera frame unavailable; stopping.")
                break
            
        # Flip the image horizontally for a natural mirror-view effect
            img = cv2.flip(img, 1)
        
        # Detect hands and draw landmarks
            img = detector.find_hands(img, draw=True)
        
        # Calculate the smoothed finger count
            finger_count = detector.count_fingers()
        
        # Apply the aesthetic UI overlay
            img = ui.draw_hud(img, finger_count, detector.gesture())
        
        # Display the result
            cv2.imshow("Real-Time Hand Finger Counter", img)
        
        # Listen for the 'q' key to quit
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
            
    # Clean up resources
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
