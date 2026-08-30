import cv2
import mediapipe as mp
import collections
import statistics

class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.7, smooth_frames=7):
        """
        Initialize the MediaPipe Hand detector.
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Landmark IDs for finger tips
        self.tip_ids = [4, 8, 12, 16, 20]
        
        # Deque for smoothing detections
        self.history = collections.deque(maxlen=max(1, smooth_frames))
        self.results = None
        self.hand_counts = {}
        self.hand_points = {}

    def find_hands(self, img, draw=True):
        """
        Detect hands in the given image.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return img

    def count_fingers(self):
        """
        Count fingers on up to two detected hands.
        Applies temporal smoothing over the last N frames.
        Returns the smoothed finger count (0-10), or None if no hand is detected.
        """
        current_count = None
        
        if self.results and self.results.multi_hand_landmarks and len(self.results.multi_handedness) > 0:
            counts = []
            self.hand_counts = {}
            self.hand_points = {}
            for hand_landmarks, hand_info in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                handedness = hand_info.classification[0].label
            
                lm_list = list(hand_landmarks.landmark)
                
                fingers = []
            
            # Thumb
            # Depending on whether it's a left or right hand, the thumb check is flipped horizontally.
                if handedness == "Right":
                    fingers.append(int(lm_list[4].x < lm_list[3].x))
                else:
                    fingers.append(int(lm_list[4].x > lm_list[3].x))
                    
            # 4 Fingers (Index, Middle, Ring, Pinky)
                for id in range(1, 5):
                # If tip y-coordinate is less than PIP y-coordinate (meaning it's higher on screen)
                    fingers.append(int(lm_list[self.tip_ids[id]].y < lm_list[self.tip_ids[id] - 2].y))
                    
                counts.append(fingers.count(1))
                self.hand_counts[handedness] = fingers.count(1)
                self.hand_points[handedness] = {"index": (lm_list[8].x, lm_list[8].y), "thumb": (lm_list[4].x, lm_list[4].y), "index_up": bool(fingers[1]), "thumb_up": bool(fingers[0])}
            current_count = sum(counts)
        
        # Add to history if hand is detected
        if current_count is not None:
            self.history.append(current_count)
        else:
            # If we want to clear the count when no hand is seen, we can clear the deque
            # Or we can just retain the previous value for a bit. Let's clear it if no hand.
            if len(self.history) > 0:
                self.history.clear()
            return None
            
        # Return mode (most common value) in history to smooth out jitter
        if len(self.history) > 0:
            try:
                smoothed_count = statistics.mode(self.history)
                return smoothed_count
            except statistics.StatisticsError:
                # If there are multiple modes, return the latest one
                return self.history[-1]
        
        return None

    def gesture(self):
        """Return a friendly gesture name based on the currently detected hands."""
        if not self.results or not self.results.multi_hand_landmarks:
            return "No hand detected"
        count = self.count_fingers()
        if len(self.results.multi_hand_landmarks) == 2 and count == 10:
            return "Both hands open"
        hand = self.results.multi_hand_landmarks[0]
        lm = list(hand.landmark)
        if count == 1 and lm[8].y < lm[6].y and lm[4].y > lm[3].y:
            return "Pointing"
        if count == 1 and lm[4].y < lm[3].y and lm[8].y > lm[6].y:
            return "Thumbs Up"
        if count == 2 and lm[8].y < lm[6].y and lm[12].y < lm[10].y:
            return "Victory / Peace"
        if count == 5:
            return "Open Hand"
        if count == 0:
            return "Fist"
        return "Special gesture"

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
