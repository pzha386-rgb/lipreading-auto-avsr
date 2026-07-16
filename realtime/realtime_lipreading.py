"""
realtime_lipreading.py — 实时唇语识别 Demo
Auto-AVSR 模型 + 摄像头 + mediapipe 人脸检测

使用方法:
    cd C:\lipread
    set KMP_DUPLICATE_LIB_OK=TRUE
    python realtime_lipreading.py

按 Q 退出, 按 SPACE 手动触发一次推理
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import argparse
import numpy as np
import cv2
import torch

# ─── mediapipe 0.10.35 Tasks API ───
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─── Auto-AVSR 模型加载 ───
# 把 auto_avsr 目录加到 path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO_AVSR_DIR = os.path.join(SCRIPT_DIR, "auto_avsr")
if AUTO_AVSR_DIR not in sys.path:
    sys.path.insert(0, AUTO_AVSR_DIR)


# ─── 嘴唇关键点索引 (mediapipe 468 点中的嘴唇部分) ───
LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
             185, 40, 39, 37, 0, 267, 269, 270, 409]
LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
             191, 80, 81, 82, 13, 312, 311, 310, 415]
LIP_ALL = list(set(LIP_OUTER + LIP_INNER))


class FaceDetector:
    """mediapipe 0.10.35 FaceLandmarker (Tasks API)"""

    def __init__(self):
        # 下载模型文件(首次运行)
        model_path = os.path.join(SCRIPT_DIR, "face_landmarker_v2_with_blendshapes.task")
        if not os.path.exists(model_path):
            print("正在下载人脸检测模型 (~30MB)...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("下载完成!")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def get_mouth_roi(self, frame_bgr, roi_size=96):
        """
        输入: BGR frame
        输出: 96x96 灰度 numpy array, 或 None
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        lms = result.face_landmarks[0]

        # 取嘴唇关键点坐标
        xs = [lms[i].x * w for i in LIP_ALL]
        ys = [lms[i].y * h for i in LIP_ALL]

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        # 嘴部区域大小 (比嘴唇范围略大)
        mouth_w = max(xs) - min(xs)
        mouth_h = max(ys) - min(ys)
        box_size = int(max(mouth_w, mouth_h) * 2.0)
        box_size = max(box_size, roi_size)

        # 裁剪
        x1 = max(0, cx - box_size // 2)
        y1 = max(0, cy - box_size // 2)
        x2 = min(w, x1 + box_size)
        y2 = min(h, y1 + box_size)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # 转灰度 + resize 到 96x96
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(gray, (roi_size, roi_size))
        return roi

    def close(self):
        self.landmarker.close()


class LipreadingModel:
    """加载 Auto-AVSR 模型做推理"""

    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        print(f"正在加载模型: {model_path}")
        print(f"设备: {self.device} (首次加载约30秒...)")

        from datamodule.data_module import DataModule
        from lightning import ModelModule

        # 用 Hydra 默认配置
        checkpoint = torch.load(model_path, map_location=self.device)

        # 构建模型
        self.model = ModelModule(cfg=None)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.loaded = True
        print("模型加载完成!")

    def predict(self, frames):
        """
        frames: list of 96x96 grayscale numpy arrays
        返回: 预测文本
        """
        if not self.loaded or len(frames) == 0:
            return ""

        # 构建 tensor: (1, T, 1, 96, 96) float32, 归一化
        video = np.stack(frames)  # (T, 96, 96)
        video = video.astype(np.float32) / 255.0
        video = torch.FloatTensor(video).unsqueeze(0).unsqueeze(2)  # (1, T, 1, 96, 96)
        video = video.to(self.device)

        with torch.no_grad():
            try:
                # Auto-AVSR 的推理接口
                output = self.model(video)
                if hasattr(output, 'text'):
                    return output.text
                elif isinstance(output, str):
                    return output
                elif isinstance(output, (list, tuple)):
                    return str(output[0]) if output else ""
                else:
                    return str(output)
            except Exception as e:
                return f"[推理错误: {e}]"


def simple_demo(model_path):
    """
    简化版 demo: 摄像头采集 → 人脸检测 → 嘴部裁剪 → 显示
    不做模型推理, 只展示 pipeline 前端部分
    """
    print("\n=== 实时唇语识别 Demo (视觉前端) ===")
    print("按 Q 退出")
    print("")

    detector = FaceDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("错误: 无法打开摄像头!")
        return

    print("摄像头已打开!")

    frame_buffer = []
    MAX_BUFFER = 75  # 3秒 @ 25fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测嘴部 ROI
        roi = detector.get_mouth_roi(frame)

        # 主画面: 显示原始帧
        display = frame.copy()

        if roi is not None:
            frame_buffer.append(roi)
            if len(frame_buffer) > MAX_BUFFER:
                frame_buffer.pop(0)

            # 右上角显示嘴部 ROI (放大到 192x192 方便看)
            roi_display = cv2.resize(roi, (192, 192))
            roi_color = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR)
            display[10:202, display.shape[1]-202:display.shape[1]-10] = roi_color

            # 状态文字
            cv2.putText(display, f"Mouth ROI: OK | Buffer: {len(frame_buffer)}/{MAX_BUFFER}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No face detected",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display, "Press Q to quit",
                   (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Lipreading Demo", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"\n总共采集了 {len(frame_buffer)} 帧嘴部 ROI")


def full_demo(model_path):
    """
    完整版 demo: 摄像头 → 人脸检测 → 嘴部裁剪 → 模型推理 → 字幕
    """
    print("\n=== 实时唇语识别 Demo (完整版) ===")
    print("按 Q 退出, 按 SPACE 手动推理")
    print("")

    detector = FaceDetector()

    # 加载模型
    try:
        model = LipreadingModel(model_path)
    except Exception as e:
        print(f"\n模型加载失败: {e}")
        print("退回到视觉前端 demo 模式...")
        detector.close()
        simple_demo(model_path)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头!")
        return

    print("摄像头已打开! 按 SPACE 触发推理\n")

    frame_buffer = []
    MAX_BUFFER = 75
    last_prediction = ""
    is_predicting = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = detector.get_mouth_roi(frame)
        display = frame.copy()

        if roi is not None:
            frame_buffer.append(roi)
            if len(frame_buffer) > MAX_BUFFER:
                frame_buffer.pop(0)

            # 右上角显示 ROI
            roi_display = cv2.resize(roi, (192, 192))
            roi_color = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR)
            display[10:202, display.shape[1]-202:display.shape[1]-10] = roi_color

            cv2.putText(display, f"Buffer: {len(frame_buffer)}/{MAX_BUFFER}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No face detected",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示预测结果
        if last_prediction:
            cv2.putText(display, f"Prediction: {last_prediction}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if is_predicting:
            cv2.putText(display, "Predicting...",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.putText(display, "SPACE=predict | Q=quit",
                   (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Lipreading Demo", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' ') and len(frame_buffer) >= 10:
            is_predicting = True
            cv2.imshow("Lipreading Demo", display)
            cv2.waitKey(1)

            print(f"推理中... ({len(frame_buffer)} 帧)")
            t0 = time.time()
            last_prediction = model.predict(list(frame_buffer))
            elapsed = time.time() - t0
            print(f"结果: {last_prediction} ({elapsed:.1f}s)")
            is_predicting = False

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


def main():
    parser = argparse.ArgumentParser(description="实时唇语识别 Demo")
    parser.add_argument("--model", type=str,
                       default="vsr_trlrs2lrs3vox2avsp_base.pth",
                       help="模型权重路径")
    parser.add_argument("--mode", type=str, default="simple",
                       choices=["simple", "full"],
                       help="simple=只显示嘴部检测, full=包含模型推理")
    args = parser.parse_args()

    if args.mode == "simple":
        simple_demo(args.model)
    else:
        full_demo(args.model)


if __name__ == "__main__":
    main()
