# tracker_amd.py
# same as tracker.py but explicitly targets AMD GPU via OpenCL
#
# AMD GPUs support OpenCL just like Intel but need the right drivers:
#   - install AMD Adrenalin drivers (includes OpenCL support)
#   - download from: https://www.amd.com/en/support
#   - opencv uses OpenCL automatically once drivers are installed
#
# optionally install ROCm for deeper GPU support (linux only mostly):
#   https://rocm.docs.amd.com
#
# to use this instead of tracker.py, change main.py import to:
#   from tracker_amd import build_hands, process_frame, frame_to_gpu_rgb

import cv2
import mediapipe as mp
import time

# AMD uses OpenCL same as Intel — just make sure AMD drivers are installed
ocl_available = cv2.ocl.haveOpenCL()
cv2.ocl.setUseOpenCL(ocl_available)

if ocl_available:
    # print which device opencv is using (should say AMD if drivers are right)
    device_name = cv2.ocl.Device.getDefault().name()
    print(f"[GPU] OpenCV OpenCL   : ENABLED  v")
    print(f"[GPU] AMD Device      : {device_name}")
else:
    print("[GPU] OpenCV OpenCL   : NOT available, falling back to CPU")
    print("[GPU] Make sure AMD Adrenalin drivers are installed")

# mediapipe gpu delegate — same situation as intel, falls back to cpu on windows
_gpu_delegate = False
try:
    from mediapipe.tasks.python.core.base_options import BaseOptions
    _test = BaseOptions(delegate=BaseOptions.Delegate.GPU)
    _gpu_delegate = True
    print("[GPU] MediaPipe AI    : GPU delegate ENABLED  v")
except Exception as e:
    print(f"[GPU] MediaPipe AI    : CPU only ({e})")

mp_hands = mp.solutions.hands


def build_hands():
    if _gpu_delegate:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        try:
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=None,
                    delegate=mp_tasks.BaseOptions.Delegate.GPU
                ),
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.7,
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                result_callback=None
            )
            return mp_vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[GPU] HandLandmarker failed ({e}), falling back to CPU")

    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )


def process_frame(hands, rgb_frame, return_timing=False):
    t0 = time.perf_counter()
    results = hands.process(rgb_frame)
    math_ms = (time.perf_counter() - t0) * 1000.0

    if not results.multi_hand_landmarks:
        landmarks = None
    else:
        h, w, _ = rgb_frame.shape
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    if return_timing:
        return landmarks, math_ms
    return landmarks


def frame_to_gpu_rgb(frame):
    # AMD uses UMat same as Intel OpenCL — opencv handles it automatically
    umat     = cv2.UMat(frame)
    umat     = cv2.flip(umat, 1)
    umat_rgb = cv2.cvtColor(umat, cv2.COLOR_BGR2RGB)
    return umat.get(), umat_rgb.get()
