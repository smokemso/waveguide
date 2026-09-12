# tracker_nvidia.py
# same as tracker.py but uses NVIDIA CUDA instead of Intel OpenCL
#
# requirements:
#   pip install opencv-contrib-python
#   needs CUDA-enabled opencv build:
#   pip uninstall opencv-python opencv-contrib-python -y
#   pip install opencv-contrib-python
#   also needs NVIDIA CUDA toolkit installed:
#   https://developer.nvidia.com/cuda-downloads
#
# to use this instead of tracker.py, change main.py import to:
#   from tracker_nvidia import build_hands, process_frame, frame_to_gpu_rgb

import cv2
import mediapipe as mp
import time

# check if CUDA is available in this opencv build
cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"[GPU] NVIDIA CUDA     : {'ENABLED  v' if cuda_available else 'NOT available, falling back to CPU'}")

if cuda_available:
    device = cv2.cuda.DeviceInfo(0)
    print(f"[GPU] CUDA Device     : {device.name()}")

# mediapipe doesnt support cuda directly so it still uses cpu for AI
# but opencv frame ops (flip, color convert) run on CUDA
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
    # same as original, mediapipe falls back to cpu on windows anyway
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
    if cuda_available:
        # upload frame to NVIDIA GPU memory
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # flip on GPU (faster than cpu)
        gpu_frame = cv2.cuda.flip(gpu_frame, 1)

        # convert BGR to RGB on GPU
        gpu_rgb = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)

        # download back to cpu for mediapipe
        frame = gpu_frame.download()
        rgb   = gpu_rgb.download()
        return frame, rgb
    else:
        # fallback to cpu if CUDA not available
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, rgb
