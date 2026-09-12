# ========================= tracker.py ========================
# Hand landmark detection via MediaPipe.
# Tries GPU delegate first, falls back to CPU automatically.
# =============================================================

import cv2
import time
import math
import numpy as np
import onnxruntime as ort
import os

# ── Intel GPU via OpenCL (OpenCV frame ops) ───────────────────
ocl_available = cv2.ocl.haveOpenCL()
cv2.ocl.setUseOpenCL(ocl_available)
print(f"[GPU] OpenCV OpenCL   : {'ENABLED  v' if ocl_available else 'NOT available, falling back to CPU'}")

_ep_active = "Unknown"
_palm_session = None
_landmark_session = None
_palm_input_name = ""
_lm_input_name = ""
_palm_shape = (192, 192)
_lm_shape = (224, 224)

def get_backend_info():
    """Return execution provider and hardware acceleration details."""
    dev_name = cv2.ocl.Device.getDefault().name() if ocl_available else "CPU only"
    return {
        "opencl_enabled": ocl_available,
        "opencl_device": dev_name,
        "mediapipe_gpu_attempted": True,
        "mediapipe_backend": f"ONNX Runtime ({_ep_active})",
    }


def build_hands():
    """Create InferenceSessions ONCE at startup."""
    global _palm_session, _landmark_session, _ep_active, _palm_input_name, _lm_input_name
    print("[GPU] Initializing ONNX Runtime Sessions...")
    
    # --- Thread Tuning for CPUExecutionProvider ---
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Heuristic for physical cores (assuming hyperthreading)
    logical_cores = os.cpu_count() or 4
    physical_cores = logical_cores // 2 if logical_cores > 2 else logical_cores
    
    options.intra_op_num_threads = physical_cores
    options.inter_op_num_threads = 1
    
    print(f"[GPU] ONNX Threading  : intra={options.intra_op_num_threads}, inter={options.inter_op_num_threads}")

    providers = ["CPUExecutionProvider"]
    provider_options = [{}]

    def load_model(path):
        try:
            sess = ort.InferenceSession(path, options, providers=providers)
        except Exception as e:
            print(f"[GPU] Error loading {os.path.basename(path)}: {e}.")
            sess = None
        return sess

    # Using the 'lite' models as MediaPipe actually uses by default
    base_path = r"C:\Users\shrad\OneDrive\Desktop\c.c\gesture_project_clean\033_Hand_Detection_and_Tracking\30_batchN_post-process_marged"
    palm_model = os.path.join(base_path, "palm_detection_lite_inf_post_192x192.onnx")
    lm_model = os.path.join(base_path, "hand_landmark_lite_1x3x224x224.onnx")

    _palm_session = load_model(palm_model)
    _landmark_session = load_model(lm_model)
    
    _ep_active = _palm_session.get_providers()[0]
    print(f"[GPU] ONNX Runtime AI   : {_ep_active} ENABLED  v")
    
    _palm_input_name = _palm_session.get_inputs()[0].name
    _lm_input_name = _landmark_session.get_inputs()[0].name

    class DummyHands:
        def process(self, rgb_frame):
            pass
    return DummyHands()


def process_frame(hands, rgb_frame, return_timing=False):
    """
    Run two-stage inference (Palm Detection -> NMS/Crop Overhead -> Hand Landmark)
    using ONNX Runtime on the Intel iGPU.
    """
    t0 = time.perf_counter()
    h, w, _ = rgb_frame.shape
    
    # === 1. Pre-process for Palm Detection (Keep Aspect Ratio + Pad) ===
    square_size = max(h, w)
    pad_h = (square_size - h) // 2
    pad_w = (square_size - w) // 2
    
    # Create padded square
    padded_square = cv2.copyMakeBorder(
        rgb_frame, pad_h, square_size - h - pad_h, pad_w, square_size - w - pad_w,
        cv2.BORDER_CONSTANT, value=[0,0,0]
    )
    
    resized_palm = cv2.resize(padded_square, _palm_shape)
    palm_tensor = np.transpose(resized_palm, (2, 0, 1))
    palm_tensor = np.expand_dims(palm_tensor.astype(np.float32) / 255.0, axis=0)
    
    # === 2. Palm Detection Inference ===
    # Output: [N, 8] -> [pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y]
    palm_out = _palm_session.run(None, {_palm_input_name: palm_tensor})
    boxes = palm_out[0]
    
    # === 3. NMS / Post-process ===
    keep = boxes[:, 0] > 0.6
    boxes = boxes[keep]
    if len(boxes) == 0:
        if return_timing: return None, (time.perf_counter() - t0) * 1000.0
        return None
        
    best_idx = np.argmax(boxes[:, 0])
    pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y = boxes[best_idx]
    
    if box_size <= 0:
        if return_timing: return None, (time.perf_counter() - t0) * 1000.0
        return None

    # === 4. Rotated Crop for Hand Landmark ===
    kp02_x = kp2_x - kp0_x
    kp02_y = kp2_y - kp0_y
    sqn_rr_size = 2.9 * box_size
    rotation = 0.5 * math.pi - math.atan2(-kp02_y, kp02_x)
    rotation = rotation - 2 * math.pi * math.floor((rotation + math.pi) / (2 * math.pi))
    
    sqn_rr_center_x = box_x + 0.5 * box_size * math.sin(rotation)
    sqn_rr_center_y = box_y - 0.5 * box_size * math.cos(rotation)
    
    center_x = sqn_rr_center_x * square_size
    center_y = sqn_rr_center_y * square_size
    crop_size = sqn_rr_size * square_size
    
    angle_deg = math.degrees(rotation)
    M = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)
    
    M[0, 2] += (crop_size / 2) - center_x
    M[1, 2] += (crop_size / 2) - center_y
    
    hand_crop = cv2.warpAffine(padded_square, M, (int(crop_size), int(crop_size)))
    
    # === 5. Hand Landmark Inference ===
    resized_lm = cv2.resize(hand_crop, _lm_shape)
    lm_tensor = np.transpose(resized_lm, (2, 0, 1))
    lm_tensor = np.expand_dims(lm_tensor.astype(np.float32) / 255.0, axis=0)
    
    lm_out = _landmark_session.run(None, {_lm_input_name: lm_tensor})
    
    # === 6. Confidence Gating & Landmark extraction ===
    confidence = float(lm_out[1][0][0])
    print(f"[DEBUG] Hand Confidence Score: {confidence:.3f}")
    
    if confidence < 0.5:
        if return_timing: return None, (time.perf_counter() - t0) * 1000.0
        return None

    landmarks_flat = lm_out[0][0]
    
    scale = 224.0 / crop_size
    M_224 = M.copy()
    M_224[0, :] *= scale
    M_224[1, :] *= scale
    
    M_inv = cv2.invertAffineTransform(M_224)
    
    landmarks = []
    # landmarks_flat gives coords in 0-224 space. Wait, PINTO scales them down to 0.0-1.0 by dividing by 224.
    # In hand_landmark.py: `rrn_lms = rrn_lms / 224`. This implies the RAW output is 0-224 pixel coords!
    # I will assume raw output is 0-224 pixel coords and apply M_inv directly.
    for i in range(21):
        px = landmarks_flat[i*3]
        py = landmarks_flat[i*3 + 1]
        
        orig_x = M_inv[0, 0] * px + M_inv[0, 1] * py + M_inv[0, 2]
        orig_y = M_inv[1, 0] * px + M_inv[1, 1] * py + M_inv[1, 2]
        
        final_x = orig_x - pad_w
        final_y = orig_y - pad_h
        
        landmarks.append((int(final_x), int(final_y)))

    math_ms = (time.perf_counter() - t0) * 1000.0

    if return_timing:
        return landmarks, math_ms
    return landmarks


def frame_to_gpu_rgb(frame):
    """
    Upload frame to GPU via UMat, flip and convert to RGB.
    Returns (bgr_frame, rgb_frame) both as CPU numpy arrays.
    """
    umat       = cv2.UMat(frame)
    umat       = cv2.flip(umat, 1)
    umat_rgb   = cv2.cvtColor(umat, cv2.COLOR_BGR2RGB)
    return umat.get(), umat_rgb.get()
