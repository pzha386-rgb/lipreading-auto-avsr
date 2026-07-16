"""
roi_utils.py — 复用 BBC eval 验证过的 mouth ROI 提取 pipeline
facexlib 检测 + 对齐 → VideoProcess 裁剪嘴部
"""
import sys
import numpy as np
import cv2
import torch
import torchvision

if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool

sys.path.insert(0, '/data/default/auto_avsr')

from facexlib.detection import init_detection_model
from facexlib.alignment import init_alignment_model
from facexlib.alignment.convert_98_to_68_landmarks import landmark_98_to_68
from preparation.detectors.retinaface.video_process import VideoProcess

_det = None
_align = None
_vp = None

def init_roi_models():
    global _det, _align, _vp
    if _det is None:
        print("Loading face detection/alignment models...")
        _det = init_detection_model("retinaface_resnet50", device="cuda")
        _align = init_alignment_model("awing_fan", device="cuda")
        _vp = VideoProcess(convert_gray=False)
        print("ROI models loaded")

def _decode_landmark(frame_rgb, bbox, margin=0.3):
    H_orig, W_orig = frame_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)/2, (y1+y2)/2
    half = max(x2-x1, y2-y1) * (1 + margin) / 2
    sx1, sy1 = max(0, int(cx-half)), max(0, int(cy-half))
    sx2, sy2 = min(W_orig, int(cx+half)), min(H_orig, int(cy+half))

    crop = frame_rgb[sy1:sy2, sx1:sx2]
    H_crop, W_crop = crop.shape[:2]
    if H_crop < 10 or W_crop < 10:
        return None

    crop_256 = cv2.resize(crop, (256, 256))
    crop_256_bgr = crop_256[..., ::-1]
    inp = torch.from_numpy(np.ascontiguousarray(
        crop_256_bgr.transpose((2, 0, 1)))).float()
    inp = inp.cuda().div(255).unsqueeze(0)

    with torch.no_grad():
        outputs, _ = _align(inp)
    heatmap = outputs[-1][:, :-1, :, :]

    hm = heatmap[0].cpu().numpy()
    flat = hm.reshape(98, -1)
    idx = flat.argmax(axis=1)
    ys = idx // 64
    xs = idx % 64
    pred = np.stack([xs, ys], axis=1).astype(np.float32)

    for i in range(98):
        py, px = ys[i], xs[i]
        if 1 <= px <= 62 and 1 <= py <= 62:
            diff_x = hm[i, py, px+1] - hm[i, py, px-1]
            diff_y = hm[i, py+1, px] - hm[i, py-1, px]
            pred[i, 0] += np.sign(diff_x) * 0.25
            pred[i, 1] += np.sign(diff_y) * 0.25

    pred = pred * 4.0
    pred = pred * np.array([W_crop / 256, H_crop / 256])
    pred = pred + np.array([sx1, sy1])
    return pred

def extract_mouth_roi_video(video_path):
    """
    输入: 任意视频路径
    输出: (T, H, W, C) numpy uint8 嘴部 ROI 视频, 或 None
    """
    init_roi_models()

    video = torchvision.io.read_video(str(video_path), pts_unit="sec")[0].numpy()
    T = video.shape[0]
    landmarks_per_frame = []
    detected = 0

    for t in range(T):
        frame_rgb = video[t]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            bboxes = _det.detect_faces(frame_bgr, 0.5)
        if len(bboxes) == 0:
            landmarks_per_frame.append(None)
            continue
        # 取最大人脸
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes]
        bbox = bboxes[int(np.argmax(areas))][:4]
        lm98 = _decode_landmark(frame_rgb, bbox)
        if lm98 is None:
            landmarks_per_frame.append(None)
            continue
        lm68 = landmark_98_to_68(lm98)
        landmarks_per_frame.append(lm68)
        detected += 1

    if detected < T * 0.5:
        raise RuntimeError(f"Face detected in only {detected}/{T} frames - video may not contain a clear face")

    # 补全缺失帧的 landmarks (用最近邻)
    last_valid = None
    for t in range(T):
        if landmarks_per_frame[t] is not None:
            last_valid = landmarks_per_frame[t]
        elif last_valid is not None:
            landmarks_per_frame[t] = last_valid
    # 前向补全开头的 None
    first_valid = next(lm for lm in landmarks_per_frame if lm is not None)
    for t in range(T):
        if landmarks_per_frame[t] is None:
            landmarks_per_frame[t] = first_valid

    roi_video = _vp(video, landmarks_per_frame)
    if roi_video is None:
        raise RuntimeError("VideoProcess returned None - landmark quality too low")
    return np.array(roi_video)
