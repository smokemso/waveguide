import cv2
import time
import math
import numpy as np
import os
import psutil

from openvino import Core

from config import HOMELAB_MODE

# ── Intel GPU via OpenCL (OpenCV frame ops) ───────────────────
ocl_available = cv2.ocl.haveOpenCL()
cv2.ocl.setUseOpenCL(ocl_available)
print(f"[GPU] OpenCV OpenCL   : {'ENABLED  v' if ocl_available else 'NOT available, falling back to CPU'}")

_core = None
_palm_compiled = None
_lm_compiled = None
_ep_active = "Unknown"
_palm_shape = (192, 192)
_lm_shape = (224, 224)

def get_backend_info():
    """Return execution provider and hardware acceleration details."""
    dev_name = cv2.ocl.Device.getDefault().name() if ocl_available else "CPU only"
    return {
        "opencl_enabled": ocl_available,
        "opencl_device": dev_name,
        "mediapipe_gpu_attempted": True,
        "mediapipe_backend": f"OpenVINO Native ({_ep_active})",
    }

def build_hands():
    """Create OpenVINO Compiled Models ONCE at startup dynamically."""
    global _core, _palm_compiled, _lm_compiled, _ep_active
    
    physical_cores = psutil.cpu_count(logical=False) or (os.cpu_count() or 4)
    logical_cores = psutil.cpu_count(logical=True) or (os.cpu_count() or 4)
    print(f"[HW] CPU: {physical_cores} physical cores, {logical_cores} logical")
    
    if HOMELAB_MODE:
        num_threads = max(2, physical_cores // 2)
        print(f"[HW] Shared-system mode: capping to {num_threads} threads to preserve headroom for other workloads, preferring GPU offload where available.")
    else:
        num_threads = physical_cores
        print(f"[HW] Dedicated mode: using {num_threads} threads for CPU inference. GPU opportunistic.")

    _core = Core()
    available = _core.available_devices
    print(f"[HW] Devices available to OpenVINO: {available}")
    
    import pathlib
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
    base_path = PROJECT_ROOT / "033_Hand_Detection_and_Tracking" / "30_batchN_post-process_marged"
    palm_model_path = base_path / "palm_detection_lite_inf_post_192x192.onnx"
    lm_model_path = base_path / "hand_landmark_lite_1x3x224x224.onnx"

    palm_model = _core.read_model(palm_model_path)
    lm_model = _core.read_model(lm_model_path)

    gpu_success = False
    target_device = "CPU"

    # Opportunistically try GPU if present
    if any(d.startswith("GPU") for d in available):
        try:
            palm_gpu = _core.compile_model(palm_model, device_name="GPU")
            lm_gpu = _core.compile_model(lm_model, device_name="GPU")
            
            # Run one dummy inference to confirm it actually executes without throwing
            dummy_input = np.zeros((1, 3, 192, 192), dtype=np.float32)
            _ = palm_gpu([dummy_input])
            
            _palm_compiled, _lm_compiled = palm_gpu, lm_gpu
            target_device = "GPU"
            gpu_success = True
            print("[HW] GPU available and verified working - using GPU")
        except Exception as e:
            print(f"[HW] GPU detected but failed verification ({e}) - falling back to CPU")

    if not gpu_success:
        print(f"[HW] Compiling CPU fallback models (Threads: {num_threads})...")
        t_compile_start = time.perf_counter()
        ov_config = {"INFERENCE_NUM_THREADS": num_threads}
        _palm_compiled = _core.compile_model(palm_model, device_name="CPU", config=ov_config)
        _lm_compiled = _core.compile_model(lm_model, device_name="CPU", config=ov_config)
        t_compile_ms = (time.perf_counter() - t_compile_start) * 1000.0
        print(f"[HW] CPU Fallback JIT Compile Time: {t_compile_ms:.2f} ms")
        target_device = "CPU"

    print(f"[HW] Selected inference device: {target_device}")
    _ep_active = target_device

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
    palm_out = _palm_compiled([palm_tensor])
    boxes = list(palm_out.values())[0]
    
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
    
    lm_out_dict = _lm_compiled([lm_tensor])
    lm_out_vals = list(lm_out_dict.values())
    
    # === 6. Confidence Gating & Landmark extraction ===
    # OpenVINO dictionary order: index 0 is (1, 63) landmarks, index 1 is (1, 1) confidence
    confidence = float(lm_out_vals[1][0][0])
    
    if confidence < 0.5:
        if return_timing: return None, (time.perf_counter() - t0) * 1000.0
        return None

    landmarks_flat = lm_out_vals[0][0]
    
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
