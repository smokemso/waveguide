# main.py
# starting point, just connects all the files and runs the loop
# run this to start the virtual mouse
#
# every frame it does:
# 1. grab frame from webcam
# 2. flip it and convert colors (gpu)
# 3. find hand and get finger positions
# 4. detect gesture and move cursor
# 5. draw dots and info on screen
# 6. show window, repeat till q is pressed
#
# ── GPU switcher ──────────────────────────────────────────────
# Intel (default) : from tracker        import ...
# NVIDIA CUDA     : from tracker_nvidia import ...
# AMD OpenCL      : from tracker_amd    import ...
# -------------------------------------------------------------

import time
import cv2
from camera   import CameraStream
from tracker  import build_hands, process_frame, frame_to_gpu_rgb, get_backend_info   # change tracker here
from gestures import GestureState, process_gestures
from hud      import draw_landmarks, draw_hud

from monitoring import MetricsLogger

def run_pipeline(enable_benchmark=False, show_debug_window=False, db_path="metrics_production.db", benchmark_frames=300):
    stream = CameraStream(index=0)  # change to 1 if wrong camera opens
    hands  = build_hands()
    state  = GestureState()
    backend_info = get_backend_info()

    device_str = backend_info.get("mediapipe_backend", "Unknown")
    logger = MetricsLogger(device=device_str, db_path=db_path)

    if enable_benchmark:
        print(f"[BENCHMARK] Starting {benchmark_frames}-frame benchmark harness (logging to SQLite)...")
    else:
        print("[INFO] Benchmark is OFF. Running normally. Press 'q' to quit.")
        
    frame_count = 0

    while True:
        # Start wall-clock timing for the entire loop iteration
        t_frame_start = time.perf_counter()

        # ── Stage 1a: Camera Read (OpenCV) ────────────────────
        t_cam_start = time.perf_counter()
        success, frame, is_new = stream.read()
        t_cam_ms = (time.perf_counter() - t_cam_start) * 1000.0

        if not success:
            continue  # skip if frame wasnt ready

        if not is_new:
            # Stale frame: skip inference and cursor logic to prevent jitter, 
            # just sleep to maintain 30fps loop cadence and try again.
            elapsed = time.perf_counter() - t_frame_start
            if elapsed < (1.0 / 30.0):
                time.sleep((1.0 / 30.0) - elapsed)
            
            # Log stale frame
            frame_count += 1
            logger.log(frame_count, opencv_ms=t_cam_ms, math_ms=0.0, total_ms=(time.perf_counter()-t_frame_start)*1000.0, is_new=False)
            
            if enable_benchmark and benchmark_frames is not None and frame_count >= benchmark_frames:
                break
                
            continue

        # ── Stage 1b: Flip and Color Conversion (OpenCV/GPU) ──
        t_conv_start = time.perf_counter()
        frame, rgb = frame_to_gpu_rgb(frame)
        t_conv_ms = (time.perf_counter() - t_conv_start) * 1000.0

        # ── Stage 2: Math/Inference Stage (MediaPipe only) ────
        landmarks, math_ms = process_frame(hands, rgb, return_timing=True)

        # ── Gesture & Cursor Logic (smoothing & cursor move) ──
        t_gesture_start = time.perf_counter()
        status = ""
        if landmarks:
            status = process_gestures(landmarks, state)
        else:
            state.reset()  # hand left frame, reset everything
        t_gesture_ms = (time.perf_counter() - t_gesture_start) * 1000.0

        # ── Stage 1c: Display & Draw Calls (OpenCV) ───────────
        t_draw_start = time.perf_counter()
        if show_debug_window:
            if landmarks:
                draw_landmarks(frame, landmarks[8], landmarks[4], landmarks[12])
            draw_hud(frame, status)
            cv2.imshow("Virtual Mouse", frame)
        key = cv2.waitKey(1) & 0xFF
        t_draw_ms = (time.perf_counter() - t_draw_start) * 1000.0

        # Total End-to-End Latency
        t_frame_end = time.perf_counter()
        total_ms = (t_frame_end - t_frame_start) * 1000.0

        # Total OpenCV Stage
        opencv_ms = t_cam_ms + t_conv_ms + t_draw_ms

        frame_count += 1
        logger.log(frame_count, opencv_ms, math_ms, total_ms, is_new=True)

        if enable_benchmark and benchmark_frames is not None and frame_count >= benchmark_frames:
            break

        if key == ord('q'):
            print("\n[INFO] Stopped early by user ('q' pressed).")
            break
            
        # ── Pacing (Frame-rate cap) ───────────────────────────
        # Ensure we don't spin faster than 30fps (~33.33ms per loop)
        elapsed = time.perf_counter() - t_frame_start
        if elapsed < (1.0 / 30.0):
            time.sleep((1.0 / 30.0) - elapsed)

    stream.stop()
    cv2.destroyAllWindows()
    logger.close()

    if enable_benchmark:
        logger.summarize()

if __name__ == "__main__":
    run_pipeline()
