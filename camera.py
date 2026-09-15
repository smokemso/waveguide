# ========================= camera.py =========================
# Threaded camera reader — cap.read() runs on background thread
# so the main loop never blocks waiting for a frame.
# =============================================================

import cv2
import threading
import time
from config import CAM_WIDTH, CAM_HEIGHT, CAM_FPS


class CameraStream:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
        print(f"[CAM] Resolution      : {CAM_WIDTH}x{CAM_HEIGHT} @ {CAM_FPS}fps")

        self.ret, self.frame = self.cap.read()
        self.lock    = threading.Lock()
        self.running = True
        self.last_cap_read_ms = 0.0
        self.new_frame = True
        self.thread  = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            t0 = time.perf_counter()
            ret, frame = self.cap.read()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with self.lock:
                self.ret, self.frame = ret, frame
                self.last_cap_read_ms = dt_ms
                self.new_frame = True

    def read(self):
        with self.lock:
            is_new = self.new_frame
            self.new_frame = False
            return self.ret, (self.frame.copy() if self.ret else None), is_new

    def get_last_cap_read_ms(self):
        with self.lock:
            return self.last_cap_read_ms

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
