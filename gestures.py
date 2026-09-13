import time
import numpy as np
import pyautogui
import threading
from config import SENSITIVITY, SMOOTHENING, CLICK_DISTANCE, SCROLL_DISTANCE, SCROLL_AMOUNT, CLICK_COOLDOWN, DEADZONE

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0  # removes 0.1s delay pyautogui adds by default (was causing lag)

screen_w, screen_h = pyautogui.size()

_click_cooldown = [0.0]  # using list so i can modify it inside functions

def _click_async(func):
    # runs click on separate thread so main loop doesnt freeze
    threading.Thread(target=func, daemon=True).start()

def _can_click():
    return time.time() - _click_cooldown[0] > CLICK_COOLDOWN

def _register_click():
    _click_cooldown[0] = time.time()

def get_distance(p1, p2):
    # normal distance formula but using numpy so its faster
    d = np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)
    return float(np.sqrt(np.dot(d, d)))

def clamp(val, lo, hi):
    # keeps value between lo and hi so cursor stays on screen
    return float(np.clip(val, lo, hi))

def smooth_cursor(dx, dy, prev_x, prev_y):
    # apply sensitivity
    delta = np.array([dx, dy], dtype=np.float32) * SENSITIVITY
    
    # deadzone filters out micro hand tremors
    delta = np.where(np.abs(delta) < DEADZONE, 0.0, delta)
    
    prev  = np.array([prev_x, prev_y], dtype=np.float32)
    smoothed = prev + (prev + delta - prev) / SMOOTHENING
    
    return (
        clamp(smoothed[0], 0, screen_w - 1),
        clamp(smoothed[1], 0, screen_h - 1)
    )

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

    # 1. EVALUATE SCROLL MODE
    # Scroll mode freezes the cursor position
    if d_index_middle < SCROLL_DISTANCE:
        state.scroll_active = True
        
        # in camera coords y=0 is at top, so index above middle = index_y < middle_y = scroll up
        if index_y < middle_y:
            pyautogui.scroll(SCROLL_AMOUNT)
            status = "Scroll Up"
        elif index_y > middle_y:
            pyautogui.scroll(-SCROLL_AMOUNT)
            status = "Scroll Down"
        else:
            pyautogui.scroll(SCROLL_AMOUNT)
            status = "Scroll Up"
    else:
        state.scroll_active = False

    # 2. EVALUATE CURSOR MOVEMENT
    # Move normally if we aren't scrolling
    if not state.scroll_active:
        if state.prev_index_x is not None:
            new_x, new_y = smooth_cursor(
                index_x - state.prev_index_x,
                index_y - state.prev_index_y,
                state.cursor_x, state.cursor_y
            )
            pyautogui.moveTo(new_x, new_y)
            state.cursor_x, state.cursor_y = new_x, new_y

    # Always update prev position so the delta is fresh
    # (even if frozen, so we don't jump when unfreezing)
    state.prev_index_x, state.prev_index_y = index_x, index_y

    # 3. EVALUATE CLICKS (INDEPENDENT)
    if d_thumb_index < CLICK_DISTANCE:
        if not state.pinch_active:
            state.pinch_active = True
            if _can_click():
                _click_async(pyautogui.click)
                _register_click()
                status = "Left Click"
    else:
        state.pinch_active = False

    if d_thumb_middle < CLICK_DISTANCE:
        if not state.rpinch_active:
            state.rpinch_active = True
            if _can_click():
                _click_async(pyautogui.rightClick)
                _register_click()
                status = "Right Click"
    else:
        state.rpinch_active = False

    return status
