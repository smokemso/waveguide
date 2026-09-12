import cv2
import time

def test_cam(name, backend=None, use_mjpg=False, resolution=(640, 480), exposure_val=None, auto_exp_val=None):
    print(f"\n--- Testing: {name} ---")
    if backend is not None:
        cap = cv2.VideoCapture(0, backend)
    else:
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("Camera failed to open.")
        return

    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
    if auto_exp_val is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exp_val)
    if exposure_val is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)

    # Warmup
    for _ in range(5): cap.read()
    
    frames = 0
    t0 = time.time()
    while time.time() - t0 < 3.0:
        ret, _ = cap.read()
        if ret:
            frames += 1
            
    elapsed = time.time() - t0
    fps = frames / elapsed
    
    # Read actual values
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    actual_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    
    print(f"Resolution: {actual_w}x{actual_h}")
    print(f"Prop FPS  : {actual_fps}")
    print(f"Auto Exp  : {actual_auto}")
    print(f"Exposure  : {actual_exp}")
    print(f"Result    : {frames} frames in {elapsed:.2f}s -> {fps:.2f} FPS")
    cap.release()

test_cam("0. Baseline (Default backend, 640x480)", resolution=(640, 480))
test_cam("1. DirectShow Backend", backend=cv2.CAP_DSHOW, resolution=(640, 480))
test_cam("2. MJPG FOURCC (Default Backend)", use_mjpg=True, resolution=(640, 480))
test_cam("3. Lower Resolution (320x240)", resolution=(320, 240))
test_cam("4. Manual Exposure (Default backend, Auto=0, Exp=-5)", resolution=(640, 480), auto_exp_val=0, exposure_val=-5)
test_cam("4b. Manual Exposure (DirectShow, Auto=0.25, Exp=-5)", backend=cv2.CAP_DSHOW, resolution=(640, 480), auto_exp_val=0.25, exposure_val=-5)
