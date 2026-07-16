"""
server.py — Auto-AVSR 推理服务 (BitaHub) v3
直接复用 ModelModule.forward() — 和 eval.py 相同的推理路径
"""
import os, sys, subprocess, tempfile, time
from argparse import Namespace

import torch

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

AUTO_AVSR_DIR = "/data/default/auto_avsr"
sys.path.insert(0, AUTO_AVSR_DIR)

WEIGHTS_PATH = "/data/default/vsr_trlrs2lrs3vox2avsp_base.pth"

app = FastAPI(title="Auto-AVSR Inference Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

import threading, uuid

TASKS = {}  # task_id -> {"status": "processing"|"done"|"error", "result": {...}}

modelmodule = None
video_transform = None
model_loaded = False

def load_model():
    global modelmodule, video_transform, model_loaded
    if model_loaded:
        return

    print("Loading Auto-AVSR model...")
    t0 = time.time()

    from lightning import ModelModule
    from datamodule.transforms import VideoTransform

    args = Namespace(
        modality="video",
        pretrained_model_path=WEIGHTS_PATH,
        transfer_frontend=False,
        transfer_encoder=False,
        ctc_weight=0.1,
    )

    m = ModelModule(args)
    if torch.cuda.is_available():
        m = m.cuda()
    m.eval()

    modelmodule = m
    video_transform = VideoTransform("test")

    print(f"Model loaded in {time.time()-t0:.1f}s")
    model_loaded = True


def rewrite_video(src, dst):
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vf", "fps=25", "-c:v", "mpeg4", "-q:v", "2", "-an", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr[-300:]}")


def load_video_tensor(video_path):
    """和 av_dataset.py 一致: read_video → (T,H,W,C) → 灰度/变换"""
    import torchvision
    vid = torchvision.io.read_video(video_path, pts_unit="sec", output_format="THWC")[0]
    # 中心裁剪成正方形再 resize 到 96x96 (适配任意输入视频)
    vid = vid.permute(0, 3, 1, 2).float()  # (T, C, H, W)
    t, c, h, w = vid.shape
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    vid = vid[:, :, y0:y0+s, x0:x0+s]
    vid = torch.nn.functional.interpolate(vid, size=(96, 96), mode="bilinear", align_corners=False)
    vid = vid.to(torch.uint8)  # (T, C, 96, 96) - VideoTransform expects TCHW
    # VideoTransform("test") 处理灰度化和归一化
    sample = video_transform(vid)
    return sample


def _process_task(task_id, raw_path, rewrite, ground_truth, extract_roi):
    try:
        load_model()
        if rewrite:
            fixed_path = raw_path.replace(".mp4", "_fixed.mp4")
            rewrite_video(raw_path, fixed_path)
            infer_path = fixed_path
        else:
            infer_path = raw_path

        if extract_roi:
            from roi_utils import extract_mouth_roi_video
            roi_frames = extract_mouth_roi_video(infer_path)
            vid = torch.from_numpy(roi_frames).permute(0, 3, 1, 2).float()
            vid = torch.nn.functional.interpolate(vid, size=(96, 96), mode="bilinear", align_corners=False)
            vid = vid.to(torch.uint8)
            sample = video_transform(vid)
        else:
            sample = load_video_tensor(infer_path)

        if torch.cuda.is_available():
            sample = sample.cuda()

        t0 = time.time()
        with torch.no_grad():
            predicted = modelmodule(sample)
        elapsed = time.time() - t0

        wer = None
        if ground_truth.strip():
            import torchaudio
            ref = ground_truth.strip().lower().split()
            hyp = predicted.strip().lower().split()
            dist = torchaudio.functional.edit_distance(ref, hyp)
            wer = round(dist / max(len(ref), 1) * 100, 2)

        try:
            os.unlink(raw_path)
            if rewrite:
                os.unlink(fixed_path)
        except:
            pass

        TASKS[task_id] = {"status": "done", "result": {
            "status": "ok",
            "prediction": predicted,
            "inference_time": round(elapsed, 2),
            "wer": wer,
            "frames": int(sample.shape[0])
        }}
    except Exception as e:
        import traceback
        TASKS[task_id] = {"status": "error", "result": {
            "status": "error", "message": str(e), "traceback": traceback.format_exc()
        }}


@app.post("/submit")
async def submit(file: UploadFile = File(...), rewrite: bool = Form(True), ground_truth: str = Form(""), extract_roi: bool = Form(False)):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        raw_path = tmp.name

    task_id = str(uuid.uuid4())[:8]
    TASKS[task_id] = {"status": "processing", "result": None}
    threading.Thread(target=_process_task, args=(task_id, raw_path, rewrite, ground_truth, extract_roi), daemon=True).start()
    return {"task_id": task_id}


@app.get("/status/{task_id}")
def task_status(task_id: str):
    task = TASKS.get(task_id)
    if task is None:
        return {"status": "not_found"}
    return {"status": task["status"], "result": task["result"]}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "weights_exist": os.path.exists(WEIGHTS_PATH),
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.post("/preload")
def preload():
    try:
        load_model()
        return {"status": "ok", "message": "Model loaded"}
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


@app.post("/infer")
async def infer(file: UploadFile = File(...), rewrite: bool = Form(True), ground_truth: str = Form(""), extract_roi: bool = Form(False)):
    try:
        load_model()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            raw_path = tmp.name

        if rewrite:
            fixed_path = raw_path.replace(".mp4", "_fixed.mp4")
            rewrite_video(raw_path, fixed_path)
            infer_path = fixed_path
        else:
            infer_path = raw_path

        if extract_roi:
            from roi_utils import extract_mouth_roi_video
            roi_frames = extract_mouth_roi_video(infer_path)  # (T,H,W,C) uint8
            vid = torch.from_numpy(roi_frames).permute(0, 3, 1, 2).float()
            vid = torch.nn.functional.interpolate(vid, size=(96, 96), mode="bilinear", align_corners=False)
            vid = vid.to(torch.uint8)
            sample = video_transform(vid)
        else:
            sample = load_video_tensor(infer_path)
        if torch.cuda.is_available():
            sample = sample.cuda()

        t0 = time.time()
        with torch.no_grad():
            predicted = modelmodule(sample)   # ModelModule.forward 完整推理
        elapsed = time.time() - t0

        wer = None
        if ground_truth.strip():
            import torchaudio
            ref = ground_truth.strip().lower().split()
            hyp = predicted.strip().lower().split()
            dist = torchaudio.functional.edit_distance(ref, hyp)
            wer = round(dist / max(len(ref), 1) * 100, 2)

        try:
            os.unlink(raw_path)
            if rewrite:
                os.unlink(fixed_path)
        except:
            pass

        return {
            "status": "ok",
            "prediction": predicted,
            "inference_time": round(elapsed, 2),
            "wer": wer,
            "frames": int(sample.shape[0])
        }
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"GPU: {torch.cuda.is_available()}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
