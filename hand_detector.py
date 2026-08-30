"""MediaPipe hand landmarks, per-hand finger counting, and gesture state."""
import collections
import statistics
import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.7, track_con=0.7, smooth_frames=7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=mode, max_num_hands=max_hands,
                                         min_detection_confidence=detection_con,
                                         min_tracking_confidence=track_con)
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.tip_ids = (4, 8, 12, 16, 20)
        self.history = collections.deque(maxlen=max(1, smooth_frames))
        self.results = None
        self.hand_counts = {}
        self.hand_points = {}
        self.last_snapshot = {"left": None, "right": None, "total": None, "gesture": "No hand detected"}

    def find_hands(self, image, draw=True):
        self.results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if draw and self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, landmarks, self.mp_hands.HAND_CONNECTIONS,
                                            self.mp_styles.get_default_hand_landmarks_style(),
                                            self.mp_styles.get_default_hand_connections_style())
        return image

    def _read_frame(self):
        self.hand_counts, self.hand_points = {}, {}
        if not self.results or not self.results.multi_hand_landmarks:
            return None
        for landmarks, handed in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
            label = handed.classification[0].label
            lm = list(landmarks.landmark)
            fingers = [int(lm[4].x < lm[3].x) if label == "Right" else int(lm[4].x > lm[3].x)]
            fingers += [int(lm[tip].y < lm[tip - 2].y) for tip in self.tip_ids[1:]]
            self.hand_counts[label] = sum(fingers)
            self.hand_points[label] = {"index": [lm[8].x, lm[8].y], "thumb": [lm[4].x, lm[4].y],
                                       "index_up": bool(fingers[1]), "thumb_up": bool(fingers[0]),
                                       "fingers": fingers}
        return sum(self.hand_counts.values())

    def snapshot(self):
        """Calculate one stable result for the current processed frame."""
        total = self._read_frame()
        if total is None:
            self.history.clear()
            self.last_snapshot = {"left": None, "right": None, "total": None, "gesture": "No hand detected"}
            return self.last_snapshot
        self.history.append(total)
        stable_total = statistics.mode(self.history)
        left, right = self.hand_counts.get("Left"), self.hand_counts.get("Right")
        if stable_total == 10 and len(self.hand_counts) == 2:
            gesture = "DRAW MODE ENABLED"
        elif stable_total == 0:
            gesture = "Fist"
        elif stable_total == 5 and len(self.hand_counts) == 1:
            gesture = "Open Hand"
        elif len(self.hand_counts) == 2:
            gesture = "Double Hand Number"
        elif self.hand_points:
            point = next(iter(self.hand_points.values()))
            gesture = "Pointing" if point["index_up"] and not point["thumb_up"] else "Special Gesture"
        else:
            gesture = "Finger Counting"
        self.last_snapshot = {"left": left, "right": right, "total": stable_total, "gesture": gesture}
        return self.last_snapshot

    def count_fingers(self):
        return self.snapshot()["total"]

    def gesture(self):
        return self.last_snapshot["gesture"]

    def close(self):
        self.hands.close()
