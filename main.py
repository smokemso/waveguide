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
import numpy as np
import cv2
from camera   import CameraStream
from tracker  import build_hands, process_frame, frame_to_gpu_rgb, get_backend_info   # change tracker here
from gestures import GestureState, process_gestures
from hud      import draw_landmarks, draw_hud

ENABLE_BENCHMARK = False
BENCHMARK_FRAMES = 300  # Only used if ENABLE_BENCHMARK is True
SHOW_DEBUG_WINDOW = False


def compute_stats(values):
    """Compute min, max, mean, median, p95, p99 for a list of timing values."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return {k: 0.0 for k in ("min", "max", "mean", "median", "p95", "p99")}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def print_summary_table(benchmark_data, detailed_data, backend_info):
    """Format and print the benchmark summary table."""
    n_frames_total = len(benchmark_data)
    if n_frames_total == 0:
        print("[BENCHMARK] No frame data recorded.")
        return

    # 1. Identify max outliers before any truncationq
    max_math = max(benchmark_data, key=lambda x: x["math_ms"])
    max_opencv = max(benchmark_data, key=lambda x: x["opencv_ms"])
    max_total = max(benchmark_data, key=lambda x: x["total_ms"])

    print("\n[OUTLIER ANALYSIS]")
    print(f"  - Max Math/Inference : {max_math['math_ms']:>6.2f} ms occurred at frame_idx {max_math['frame_idx']}")
    print(f"  - Max OpenCV Stage   : {max_opencv['opencv_ms']:>6.2f} ms occurred at frame_idx {max_opencv['frame_idx']}")
    print(f"  - Max End-to-End     : {max_total['total_ms']:>6.2f} ms occurred at frame_idx {max_total['frame_idx']}")

    # 2. Exclude warm-up frames (e.g. first 10)
    WARMUP = 10
    if n_frames_total > WARMUP:
        print(f"\n[INFO] Excluding the first {WARMUP} warm-up frames from statistics to ensure steady-state accuracy.")
        benchmark_data = benchmark_data[WARMUP:]
        detailed_data = detailed_data[WARMUP:]
    
    n_frames = len(benchmark_data)
    if n_frames == 0:
        return

    # Extract the 3 required stages
    opencv_vals = [d["opencv_ms"] for d in benchmark_data]
    math_vals   = [d["math_ms"] for d in benchmark_data]
    total_vals  = [d["total_ms"] for d in benchmark_data]

    stats_opencv = compute_stats(opencv_vals)
    stats_math   = compute_stats(math_vals)
    stats_total  = compute_stats(total_vals)

    # Sub-stage statistics
    stats_cam_read    = compute_stats([d["cam_read_ms"] for d in detailed_data])
    stats_color_conv  = compute_stats([d["color_conv_ms"] for d in detailed_data])
    stats_draw_disp   = compute_stats([d["draw_display_ms"] for d in detailed_data])
    stats_cursor      = compute_stats([d["gesture_cursor_ms"] for d in detailed_data])
    stats_cap_hw      = compute_stats([d["cap_read_ms"] for d in detailed_data])

    # Overall metrics for the steady-state period
    total_wall_time = sum(total_vals) / 1000.0  # approximate wall time of remaining frames
    avg_fps = n_frames / total_wall_time if total_wall_time > 0 else 0.0
    jank_frames = sum(1 for d in benchmark_data if d["total_ms"] > 33.0)
    jank_pct = (jank_frames / n_frames) * 100.0
    
    stale_frames = sum(1 for d in benchmark_data if not d.get("is_new", True))
    stale_pct = (stale_frames / n_frames) * 100.0

    print("\n" + "=" * 88)
    print(f"               STEADY-STATE BENCHMARK SUMMARY ({n_frames} FRAMES)")
    print("=" * 88)
    print("Hardware & Execution Provider Details:")
    print(f"   OpenCV OpenCL        : {'ENABLED' if backend_info.get('opencl_enabled') else 'DISABLED'} (Device: {backend_info.get('opencl_device', 'N/A')})")
    print(f"   AI Inference Engine  : {backend_info.get('mediapipe_backend', 'CPU')}")
    print(f"   Execution Provider   : {backend_info.get('mediapipe_backend', 'CPU')}")
    print(f"   GPU EP Confirmation  : MediaPipe GPU Delegate unsupported on Windows -> Running on CPU")
    print("-" * 88)
    print(f"{'Stage':<32} {'Min (ms)':>9} {'Max (ms)':>9} {'Mean (ms)':>10} {'Median (ms)':>12} {'P95 (ms)':>9} {'P99 (ms)':>9}")
    print("-" * 88)

    row_fmt = "{:<32} {:>9.2f} {:>9.2f} {:>10.2f} {:>12.2f} {:>9.2f} {:>9.2f}"
    print(row_fmt.format("1. OpenCV Stage", stats_opencv["min"], stats_opencv["max"], stats_opencv["mean"], stats_opencv["median"], stats_opencv["p95"], stats_opencv["p99"]))
    print(row_fmt.format("2. Math/Inference Stage", stats_math["min"], stats_math["max"], stats_math["mean"], stats_math["median"], stats_math["p95"], stats_math["p99"]))
    print(row_fmt.format("3. End-to-End Latency", stats_total["min"], stats_total["max"], stats_total["mean"], stats_total["median"], stats_total["p95"], stats_total["p99"]))

    print("-" * 88)
    print("Detailed Sub-Stage Breakdown (for reference):")
    print(row_fmt.format("  - Camera Read (stream.read)", stats_cam_read["min"], stats_cam_read["max"], stats_cam_read["mean"], stats_cam_read["median"], stats_cam_read["p95"], stats_cam_read["p99"]))
    print(row_fmt.format("  - Flip & Color Conv (GPU)", stats_color_conv["min"], stats_color_conv["max"], stats_color_conv["mean"], stats_color_conv["median"], stats_color_conv["p95"], stats_color_conv["p99"]))
    print(row_fmt.format("  - Display/Draw (imshow/HUD)", stats_draw_disp["min"], stats_draw_disp["max"], stats_draw_disp["mean"], stats_draw_disp["median"], stats_draw_disp["p95"], stats_draw_disp["p99"]))
    print(row_fmt.format("  - Cursor/Gesture Logic", stats_cursor["min"], stats_cursor["max"], stats_cursor["mean"], stats_cursor["median"], stats_cursor["p95"], stats_cursor["p99"]))
    print(row_fmt.format("  - HW Frame Grab (cap.read)", stats_cap_hw["min"], stats_cap_hw["max"], stats_cap_hw["mean"], stats_cap_hw["median"], stats_cap_hw["p95"], stats_cap_hw["p99"]))

    print("-" * 88)
    print("Pipeline Performance Overview:")
    print(f"   Total Steady-State Dur   : {total_wall_time:.2f} seconds")
    print(f"   Total Frames Processed   : {n_frames} / {n_frames_total}")
    print(f"   Average Measured FPS     : {avg_fps:.2f} FPS (Frame budget @ 30 FPS: 33.33 ms)")
    print(f"   Jank Frames (>33ms total): {jank_frames} / {n_frames} ({jank_pct:.1f}%)")
    print(f"   Stale Frames (Redundant) : {stale_frames} / {n_frames} ({stale_pct:.1f}%)")
    print("=" * 88 + "\n")


from monitoring import MetricsLogger

def main():
    stream = CameraStream(index=0)  # change to 1 if wrong camera opens
    hands  = build_hands()
    state  = GestureState()
    backend_info = get_backend_info()

    device_str = backend_info.get("mediapipe_backend", "Unknown")
    logger = MetricsLogger(device=device_str, db_path="metrics_production.db")

    print("[INFO] Running in production mode. Metrics are being logged to SQLite in the background. Press 'q' to quit.")
        
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
        if SHOW_DEBUG_WINDOW:
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

if __name__ == "__main__":
    main()
