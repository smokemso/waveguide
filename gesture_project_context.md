# Gesture Tracking / Virtual Mouse Project

This document contains all source files for the Hand Gesture Tracking & Virtual Mouse system.

## Project File Structure
```text
gesture_project/
├── main.py
├── main_1.py
├── gestures.py
├── camera.py
├── hud.py
├── config.py
├── tracker.py
├── tracker_amd.py
├── tracker_nvidia.py
├── project_iteration_1.py
```

## File: `main.py`
```python
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

import cv2
from camera   import CameraStream
from tracker  import build_hands, process_frame, frame_to_gpu_rgb   # change tracker here
from gestures import GestureState, process_gestures
from hud      import draw_landmarks, draw_hud


def main():
    stream = CameraStream(index=0)  # change to 1 if wrong camera opens
    hands  = build_hands()
    state  = GestureState()

    while True:
        success, frame = stream.read()
        if not success:
            continue  # skip if frame wasnt ready

        frame, rgb = frame_to_gpu_rgb(frame)
        landmarks  = process_frame(hands, rgb)

        status = ""
        if landmarks:
            status = process_gestures(landmarks, state)
            draw_landmarks(frame, landmarks[8], landmarks[4], landmarks[12])
        else:
            state.reset()  # hand left frame, reset everything

        draw_hud(frame, status)
        cv2.imshow("Virtual Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

```
---

## File: `main_1.py`
```python
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

```
---

## File: `gestures.py`
```python
# gestures.py
# handles cursor movement, left click, right click and scroll
# left click  = pinch index + thumb
# right click = pinch middle + thumb
# scroll      = bring index + middle close together
# only one gesture fires at a time so they dont mess each other up

import numpy as np
import pyautogui
import threading
import time

from config import (
    SENSITIVITY, SMOOTHENING, DEADZONE,
    CLICK_DISTANCE, SCROLL_DISTANCE, SCROLL_AMOUNT, CLICK_COOLDOWN
)

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0  # removes 0.1s delay pyautogui adds by default (was causing lag)

screen_w, screen_h = pyautogui.size()


def get_distance(p1, p2):
    # normal distance formula but using numpy so its faster
    d = np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)
    return float(np.sqrt(np.dot(d, d)))


def clamp(val, lo, hi):
    # keeps value between lo and hi so cursor stays on screen
    return float(np.clip(val, lo, hi))


def smooth_cursor(dx, dy, prev_x, prev_y):
    # takes finger movement and converts it to smooth cursor movement
    # 1. multiply by sensitivity so small movements still move cursor
    # 2. ignore if movement is too small (deadzone) - fixes hand tremor
    # 3. dont jump straight to target, ease towards it (smoothing)
    delta = np.array([dx, dy], dtype=np.float32) * SENSITIVITY
    delta = np.where(np.abs(delta) < DEADZONE, 0.0, delta)
    prev  = np.array([prev_x, prev_y], dtype=np.float32)
    smoothed = prev + (prev + delta - prev) / SMOOTHENING
    return (
        clamp(smoothed[0], 0, screen_w - 1),
        clamp(smoothed[1], 0, screen_h - 1)
    )


_click_cooldown = [0.0]  # using list so i can modify it inside functions


def _click_async(func):
    # runs click on separate thread so main loop doesnt freeze
    threading.Thread(target=func, daemon=True).start()


def _can_click():
    return time.time() - _click_cooldown[0] > CLICK_COOLDOWN


def _register_click():
    _click_cooldown[0] = time.time()


class GestureState:
    # stores whats happening between frames
    # needed so clicks dont fire 30 times a second while pinching
    def __init__(self):
        self.prev_index_x  = None
        self.prev_index_y  = None
        self.cursor_x      = screen_w // 2
        self.cursor_y      = screen_h // 2
        self.pinch_active  = False   # is left click pinch being held?
        self.rpinch_active = False   # is right click pinch being held?
        self.scroll_active = False

    def reset(self):
        # called when hand disappears from frame
        self.prev_index_x  = None
        self.prev_index_y  = None
        self.pinch_active  = False
        self.rpinch_active = False
        self.scroll_active = False


def process_gestures(landmarks, state: GestureState):
    # main function, runs every frame
    # moves cursor and checks for gestures, returns label for hud

    index  = landmarks[8]   # index tip
    thumb  = landmarks[4]   # thumb tip
    middle = landmarks[12]  # middle tip

    index_x,  index_y  = index
    middle_x, middle_y = middle

    status = ""

    # move cursor based on how much index finger moved since last frame
    if state.prev_index_x is not None:
        new_x, new_y = smooth_cursor(
            index_x - state.prev_index_x,
            index_y - state.prev_index_y,
            state.cursor_x, state.cursor_y
        )
        pyautogui.moveTo(new_x, new_y)
        state.cursor_x, state.cursor_y = new_x, new_y

    state.prev_index_x, state.prev_index_y = index_x, index_y

    # measure distances between finger tips
    d_thumb_index  = get_distance(thumb, index)
    d_thumb_middle = get_distance(thumb, middle)
    d_index_middle = get_distance(index, middle)

    # using if/elif so only one gesture fires at a time
    if d_thumb_index < CLICK_DISTANCE:
        # left click
        state.rpinch_active = False
        state.scroll_active = False
        if not state.pinch_active:
            state.pinch_active = True
            if _can_click():
                _click_async(pyautogui.click)
                _register_click()
                status = "Left Click"

    elif d_thumb_middle < CLICK_DISTANCE:
        # right click
        state.pinch_active  = False
        state.scroll_active = False
        if not state.rpinch_active:
            state.rpinch_active = True
            if _can_click():
                _click_async(pyautogui.rightClick)
                _register_click()
                status = "Right Click"

    elif d_index_middle < SCROLL_DISTANCE:
        # scroll - index above middle = up, below = down
        state.pinch_active  = False
        state.rpinch_active = False
        if index_y < middle_y:
            pyautogui.scroll(SCROLL_AMOUNT)
            status = "Scroll Up"
        else:
            pyautogui.scroll(-SCROLL_AMOUNT)
            status = "Scroll Down"

    else:
        # no gesture, reset everything
        state.pinch_active  = False
        state.rpinch_active = False
        state.scroll_active = False

    return status

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0  # removes 0.1s delay pyautogui adds by default (was causing lag)

screen_w, screen_h = pyautogui.size()


def get_distance(p1, p2):
    # normal distance formula but using numpy so its faster
    d = np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)
    return float(np.sqrt(np.dot(d, d)))


def clamp(val, lo, hi):
    # keeps value between lo and hi so cursor stays on screen
    return float(np.clip(val, lo, hi))


def smooth_cursor(dx, dy, prev_x, prev_y):
    # if total movement is less than 3x3 pixels, dont move cursor at all
    # this kills hand tremor without affecting real movement
    if abs(dx) < 3 and abs(dy) < 3:
        return prev_x, prev_y

    # apply sensitivity
    delta = np.array([dx, dy], dtype=np.float32) * SENSITIVITY

def smooth_cursor(dx, dy, prev_x, prev_y):
    delta = np.array([dx, dy], dtype=np.float32) * SENSITIVITY
    delta = np.where(np.abs(delta) < DEADZONE, 0.0, delta)
    prev  = np.array([prev_x, prev_y], dtype=np.float32)
    smoothed = prev + (prev + delta - prev) / SMOOTHENING
    return (
        clamp(smoothed[0], 0, screen_w - 1),
        clamp(smoothed[1], 0, screen_h - 1)
    )


_click_cooldown = [0.0]  # using list so i can modify it inside functions


def _click_async(func):
    # runs click on separate thread so main loop doesnt freeze
    threading.Thread(target=func, daemon=True).start()


def _can_click():
    return time.time() - _click_cooldown[0] > CLICK_COOLDOWN


def _register_click():
    _click_cooldown[0] = time.time()


class GestureState:
    # stores whats happening between frames
    # needed so clicks dont fire 30 times a second while pinching
    def __init__(self):
        self.prev_index_x  = None
        self.prev_index_y  = None
        self.cursor_x      = screen_w // 2
        self.cursor_y      = screen_h // 2
        self.pinch_active  = False   # is left click pinch being held?
        self.rpinch_active = False   # is right click pinch being held?
        self.scroll_active = False

    def reset(self):
        # called when hand disappears from frame
        self.prev_index_x  = None
        self.prev_index_y  = None
        self.pinch_active  = False
        self.rpinch_active = False
        self.scroll_active = False


def process_gestures(landmarks, state: GestureState):
    # main function, runs every frame
    # moves cursor and checks for gestures, returns label for hud

    index  = landmarks[8]   # index tip
    thumb  = landmarks[4]   # thumb tip
    middle = landmarks[12]  # middle tip

    index_x,  index_y  = index
    middle_x, middle_y = middle

    status = ""

    # measure distances between finger tips
    d_thumb_index  = get_distance(thumb, index)
    d_thumb_middle = get_distance(thumb, middle)
    d_index_middle = get_distance(index, middle)

    # using if/elif so only one gesture fires at a time
    if d_thumb_index < CLICK_DISTANCE:
        # left click
        state.rpinch_active = False
        state.scroll_active = False
        if not state.pinch_active:
            state.pinch_active = True
            if _can_click():
                _click_async(pyautogui.click)
                _register_click()
                status = "Left Click"

        # cursor moves normally during clicks
        if state.prev_index_x is not None:
            new_x, new_y = smooth_cursor(
                index_x - state.prev_index_x,
                index_y - state.prev_index_y,
                state.cursor_x, state.cursor_y
            )
            pyautogui.moveTo(new_x, new_y)
            state.cursor_x, state.cursor_y = new_x, new_y

    elif d_thumb_middle < CLICK_DISTANCE:
        # right click
        state.pinch_active  = False
        state.scroll_active = False
        if not state.rpinch_active:
            state.rpinch_active = True
            if _can_click():
                _click_async(pyautogui.rightClick)
                _register_click()
                status = "Right Click"

        # cursor moves normally during right click
        if state.prev_index_x is not None:
            new_x, new_y = smooth_cursor(
                index_x - state.prev_index_x,
                index_y - state.prev_index_y,
                state.cursor_x, state.cursor_y
            )
            pyautogui.moveTo(new_x, new_y)
            state.cursor_x, state.cursor_y = new_x, new_y

    elif d_index_middle < SCROLL_DISTANCE:
        # scroll mode — cursor is LOCKED, doesnt move while scrolling
        state.pinch_active  = False
        state.rpinch_active = False
        state.scroll_active = True

        # in camera coords y=0 is at top, so index above middle = index_y < middle_y = scroll up
        if index_y < middle_y:
            pyautogui.scroll(SCROLL_AMOUNT)
            status = "Scroll Up"
        elif index_y > middle_y:
            pyautogui.scroll(-SCROLL_AMOUNT)
            status = "Scroll Down"
        else:
            # fingers exactly level, default to scroll up (priority)
            pyautogui.scroll(SCROLL_AMOUNT)
            status = "Scroll Up"

        # cursor stays frozen — dont update prev position either
        # so when scroll breaks, cursor doesnt jump
        state.prev_index_x, state.prev_index_y = index_x, index_y
        return status

    else:
        # no gesture, reset everything
        state.pinch_active  = False
        state.rpinch_active = False
        state.scroll_active = False

        # cursor moves normally
        if state.prev_index_x is not None:
            new_x, new_y = smooth_cursor(
                index_x - state.prev_index_x,
                index_y - state.prev_index_y,
                state.cursor_x, state.cursor_y
            )
            pyautogui.moveTo(new_x, new_y)
            state.cursor_x, state.cursor_y = new_x, new_y

    state.prev_index_x, state.prev_index_y = index_x, index_y
    return status

```
---

## File: `camera.py`
```python
# ========================= camera.py =========================
# Threaded camera reader — cap.read() runs on background thread
# so the main loop never blocks waiting for a frame.
# =============================================================

import cv2
import threading
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
        self.thread  = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

```
---

## File: `hud.py`
```python
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

```
---

## File: `config.py`
```python
# config.py
# all settings are here, change stuff here only

# cursor stuff
SENSITIVITY    = 40    # higher = moves faster
SMOOTHENING    = 10    # higher = less shaky
DEADZONE       = 8     # ignores tiny movements (hand tremor fix)

# clicking and scrolling
CLICK_DISTANCE  = 40   # how close fingers need to be for a pinch
SCROLL_DISTANCE = 40   # same but for scrolling
SCROLL_AMOUNT   = 15   # how fast it scrolls
CLICK_COOLDOWN  = 0.4  # gap between clicks so it doesnt spam

# camera
CAM_WIDTH  = 640
CAM_HEIGHT = 480
CAM_FPS    = 30

```
---

## File: `tracker.py`
```python
# ========================= tracker.py ========================
# Hand landmark detection via MediaPipe.
# Tries GPU delegate first, falls back to CPU automatically.
# =============================================================

import cv2
import mediapipe as mp

# ── Intel GPU via OpenCL (OpenCV frame ops) ───────────────────
ocl_available = cv2.ocl.haveOpenCL()
cv2.ocl.setUseOpenCL(ocl_available)
print(f"[GPU] OpenCV OpenCL   : {'ENABLED  v' if ocl_available else 'NOT available, falling back to CPU'}")

# ── MediaPipe GPU Delegate ────────────────────────────────────
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
    """Return a MediaPipe Hands instance (GPU if available, else CPU)."""
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


def process_frame(hands, rgb_frame):
    """
    Run hand detection on an RGB frame.
    Returns list of landmark (x, y) pixel tuples for the first hand,
    or None if no hand detected.
    """
    results = hands.process(rgb_frame)
    if not results.multi_hand_landmarks:
        return None

    h, w, _ = rgb_frame.shape
    hand_landmarks = results.multi_hand_landmarks[0]
    return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]


def frame_to_gpu_rgb(frame):
    """
    Upload frame to GPU via UMat, flip and convert to RGB.
    Returns (bgr_frame, rgb_frame) both as CPU numpy arrays.
    """
    umat       = cv2.UMat(frame)
    umat       = cv2.flip(umat, 1)
    umat_rgb   = cv2.cvtColor(umat, cv2.COLOR_BGR2RGB)
    return umat.get(), umat_rgb.get()

```
---

## File: `tracker_amd.py`
```python
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


def process_frame(hands, rgb_frame):
    results = hands.process(rgb_frame)
    if not results.multi_hand_landmarks:
        return None

    h, w, _ = rgb_frame.shape
    hand_landmarks = results.multi_hand_landmarks[0]
    return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]


def frame_to_gpu_rgb(frame):
    # AMD uses UMat same as Intel OpenCL — opencv handles it automatically
    umat     = cv2.UMat(frame)
    umat     = cv2.flip(umat, 1)
    umat_rgb = cv2.cvtColor(umat, cv2.COLOR_BGR2RGB)
    return umat.get(), umat_rgb.get()

```
---

## File: `tracker_nvidia.py`
```python
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


def process_frame(hands, rgb_frame):
    # same as original
    results = hands.process(rgb_frame)
    if not results.multi_hand_landmarks:
        return None

    h, w, _ = rgb_frame.shape
    hand_landmarks = results.multi_hand_landmarks[0]
    return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]


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

```
---

## File: `project_iteration_1.py`
```python
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time

# ================= SETTINGS =================
SENSITIVITY = 1.5          # 🔥 Change this to adjust cursor speed
SMOOTHENING = 1            # As requested
CLICK_DISTANCE = 30        # Pinch distance threshold
DOUBLE_PINCH_TIME = 0.4    # Time for double pinch detection
# ============================================
def cursor_movement (index, x, screen):
    a=np.interp(index, (0, x), (0, screen))
    a*=SENSITIVITY
    return(a)
def pinch(distance_thumb_index,CLICK_DISTANCE):
    
    pass
    
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
last_pinch_time = 0
pinch_count = 0
hand_present = False

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_present = True

        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append((int(lm.x * w), int(lm.y * h)))

            # Index finger tip
            index_x, index_y = landmarks[8]

            # Thumb tip
            thumb_x, thumb_y = landmarks[4]

            # Middle finger tip
            middle_x, middle_y = landmarks[12]

            # ================= CURSOR MOVEMENT =================
            screen_x=cursor_movement(index_x,w,screen_w)
            screen_y =cursor_movement(index_y,h,screen_h)
            # Clamp inside screen
            screen_x = max(0, min(screen_w - 1, screen_x))
            screen_y = max(0, min(screen_h - 1, screen_y))

            # Smoothening
            curr_x = prev_x + (screen_x - prev_x) / SMOOTHENING
            curr_y = prev_y + (screen_y - prev_y) / SMOOTHENING

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            # ================= PINCH DETECTION =================
            distance_thumb_index = math.hypot(index_x - thumb_x,
                                              index_y - thumb_y)

            if distance_thumb_index < CLICK_DISTANCE:

                current_time = time.time()

                if current_time - last_pinch_time < DOUBLE_PINCH_TIME:
                    pinch_count += 1
                else:
                    pinch_count = 1

                last_pinch_time = current_time

                # Double pinch → Right Click
                if pinch_count == 2:
                    pyautogui.rightClick()
                    cv2.putText(frame, "Right Click", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    pinch_count = 0
                    time.sleep(0.3)

                # Single pinch → Left Click
                elif pinch_count == 1:
                    pyautogui.click()
                    cv2.putText(frame, "Left Click", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    time.sleep(0.2)

            # ================= SCROLL (Two Fingers) =================
            distance_index_middle = math.hypot(index_x - middle_x,
                                               index_y - middle_y)

            if distance_index_middle < 40:
                pyautogui.scroll(30)
                cv2.putText(frame, "Scroll", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 0), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    else:
        # Hand not present → just freeze cursor
        hand_present = False

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
---
