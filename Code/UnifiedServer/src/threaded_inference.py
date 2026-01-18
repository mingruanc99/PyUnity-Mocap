import threading
import cv2
import mediapipe as mp
import time

class ThreadedInference:
    def __init__(self, face_landmarker, hand_landmarker):
        self.face_landmarker = face_landmarker
        self.hand_landmarker = hand_landmarker
        self.face_results = None
        self.hand_results = None
        self.thread = None
        self.lock = threading.Lock()
        self.stopped = False

    def start(self, image_rgb):
        self.stopped = False
        self.thread = threading.Thread(target=self.run, args=(image_rgb,))
        self.thread.daemon = True
        self.thread.start()

    def run(self, image_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int(time.time() * 1000)

        face_results = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_results = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        with self.lock:
            self.face_results = face_results
            self.hand_results = hand_results

    def get_results(self):
        with self.lock:
            return self.face_results, self.hand_results

    def stop(self):
        self.stopped = True
