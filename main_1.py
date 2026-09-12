# ========================= main.py ===========================
# Entry point. Wires all modules together.
# Run this file to start the virtual mouse.
# =============================================================

import cv2
from camera   import CameraStream
from tracker  import build_hands, process_frame, frame_to_gpu_rgb
from gestures import GestureState, process_gestures
from hud      import draw_landmarks, draw_hud


def main():
    stream = CameraStream(index=0)
    hands  = build_hands()
    state  = GestureState()

    while True:
        success, frame = stream.read()
        if not success:
            continue

        # GPU-accelerated flip + color convert
        frame, rgb = frame_to_gpu_rgb(frame)

        # Hand detection
        landmarks = process_frame(hands, rgb)

        status = ""
        if landmarks:
            status = process_gestures(landmarks, state)
            draw_landmarks(frame,
                           landmarks[8],   # index
                           landmarks[4],   # thumb
                           landmarks[12])  # middle
        else:
            state.reset()

        # HUD overlay
        draw_hud(frame, status)

        cv2.imshow("Virtual Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
