# ========================= hud.py ============================
# Draws FPS, GPU status, and gesture labels onto the frame.
# =============================================================

import cv2
import time
from tracker import ocl_available
from config  import SENSITIVITY, SMOOTHENING

_fps_state = [time.time()]


def draw_landmarks(frame, index, thumb, middle):
    """Draw 3 key finger dots and pinch line (faster than full skeleton)."""
    for pt in [index, thumb, middle]:
        cv2.circle(frame, pt, 6, (0, 255, 255), -1)
    cv2.line(frame, thumb, index, (255, 128, 0), 2)


def draw_hud(frame, status_text):
    """Draw FPS, GPU label, gesture status and settings onto frame."""
    h, w, _ = frame.shape

    # FPS
    curr_time      = time.time()
    fps            = 1.0 / max(curr_time - _fps_state[0], 1e-6)
    _fps_state[0]  = curr_time

    # GPU label
    gpu_label  = "GPU: OpenCL ON" if ocl_available else "GPU: CPU only"
    gpu_color  = (0, 255, 0)      if ocl_available else (0, 0, 255)

    cv2.putText(frame, f"FPS: {int(fps)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, gpu_label,
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gpu_color, 2)
    cv2.putText(frame, f"Sens:{SENSITIVITY} Smooth:{SMOOTHENING}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    if status_text:
        cv2.putText(frame, status_text, (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 128), 2)
