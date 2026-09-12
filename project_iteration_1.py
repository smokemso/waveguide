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