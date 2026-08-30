import cv2
import time
import numpy as np

# A vibrant, attractive palette mapped to each number (0-5)
# Format: BGR for OpenCV
COLORS = {
    0: (75, 75, 255),    # Soft Coral / Red (BGR)
    1: (3, 183, 255),    # Bright Amber (BGR)
    2: (255, 229, 0),    # Electric Cyan (BGR)
    3: (160, 214, 6),    # Emerald Green (BGR)
    4: (221, 78, 157),   # Vivid Violet (BGR)
    5: (133, 37, 247),   # Neon Pink (BGR)
    None: (200, 200, 200) # Gray if no hand detected
}

class UIOverlay:
    def __init__(self):
        self.p_time = 0
        
    def draw_hud(self, img, finger_count, gesture="", label=None):
        """
        Draw a modern, visually attractive overlay on the camera frame.
        """
        h, w, _ = img.shape
        
        # Determine current primary color
        color = COLORS.get(finger_count, COLORS[None])
        
        # Draw translucent background card (Top-Left)
        overlay = img.copy()
        card_w, card_h = 360, 175
        cv2.rectangle(overlay, (20, 20), (20 + card_w, 20 + card_h), (20, 20, 20), -1)
        # Apply semi-transparency
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Draw Card Border
        cv2.rectangle(img, (20, 20), (20 + card_w, 20 + card_h), color, 2, cv2.LINE_AA)
        
        # Finger Count Large Text
        display_text = label if label is not None else (str(finger_count) if finger_count is not None else "-")
        # Center the large number visually in the left part of the card
        cv2.putText(img, display_text, (40, 140), cv2.FONT_HERSHEY_DUPLEX, 4, color, 8, cv2.LINE_AA)
        cv2.putText(img, display_text, (40, 140), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Status Text (Right part of the card)
        cv2.putText(img, "Fingers", (140, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Detected", (140, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        status = "Active" if finger_count is not None else "Searching..."
        status_color = (100, 255, 100) if finger_count is not None else (150, 150, 150)
        cv2.putText(img, gesture[:22], (140, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(img, status, (140, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
        
        # Draw FPS indicator at bottom right
        c_time = time.time()
        instant_fps = 0 if not self.p_time else 1 / max(c_time - self.p_time, 1e-6)
        self.p_time = c_time
        self.fps = getattr(self, "fps", 0) * 0.9 + instant_fps * 0.1
        
        fps_text = f"FPS: {round(self.fps)}"
        cv2.putText(img, fps_text, (w - 120, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Instructions overlay
        instruction = "Press 'Q' to Exit"
        cv2.putText(img, instruction, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        return img
