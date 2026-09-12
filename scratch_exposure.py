import cv2
import time

def test_config(name, backend, exposure_val=None):
    print(f"\n--- Testing: {name} ---")
    
    # Use index 0, but explicitly provide the backend enum
    if backend == cv2.CAP_ANY:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0, backend)
        
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if exposure_val is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, exposure_val)
        actual_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        print(f"Requested AutoExposure={exposure_val}, Readback={actual_exp}")

    # Let the camera auto-expose/warm-up for ~2 seconds
    t_end = time.time() + 2.0
    frames_read = 0
    t_start = time.time()
    
    while time.time() < t_end:
        ret, frame = cap.read()
        if ret: frames_read += 1
        
    # Grab 5 frames to average the mean
    means = []
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            means.append(frame.mean())
            
    if means:
        avg_mean = sum(means) / len(means)
        fps = frames_read / (time.time() - t_start)
        print(f"Capture Rate during warmup: {fps:.1f} FPS")
        print(f"Result -> Mean Pixel Value: {avg_mean:.2f}")
    else:
        print("Result -> Failed to read frames.")
        
    cap.release()

if __name__ == "__main__":
    test_config("MSMF (Default)", cv2.CAP_MSMF)
    test_config("MSMF + AutoExposure=1 (Manual/Custom)", cv2.CAP_MSMF, 1.0)
    test_config("MSMF + AutoExposure=0.75 (Auto)", cv2.CAP_MSMF, 0.75)
    test_config("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW)
