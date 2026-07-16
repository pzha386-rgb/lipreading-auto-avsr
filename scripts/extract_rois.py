"""
extract_rois.py — BBC test 集 mouth ROI 提取
"""
import sys, time, csv
from pathlib import Path
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


def decode_landmark(align, frame_rgb, bbox, margin=0.3):
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
        outputs, _ = align(inp)
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


def process_clip(clip_path, det, align, video_process):
    video = torchvision.io.read_video(str(clip_path), pts_unit="sec")[0].numpy()
    T = video.shape[0]
    landmarks_per_frame = []
    detected_count = 0
    
    for t in range(T):
        frame_rgb = video[t]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            bboxes = det.detect_faces(frame_bgr, 0.5)
        if len(bboxes) == 0:
            landmarks_per_frame.append(None)
            continue
        best = max(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        bbox = [float(best[0]), float(best[1]), float(best[2]), float(best[3])]
        lm98 = decode_landmark(align, frame_rgb, bbox)
        if lm98 is None:
            landmarks_per_frame.append(None)
            continue
        lm68 = landmark_98_to_68(lm98)
        landmarks_per_frame.append(lm68)
        detected_count += 1
    
    if detected_count == 0:
        return None, 0, T
    rois = video_process(video, landmarks_per_frame)
    return rois, detected_count, T


def write_grayscale_mp4(rois, out_path, fps=25):
    T, H, W = rois.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H), isColor=False)
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H), isColor=True)
        for t in range(T):
            writer.write(cv2.cvtColor(rois[t], cv2.COLOR_GRAY2BGR))
    else:
        for t in range(T):
            writer.write(rois[t])
    writer.release()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", required=True)
    parser.add_argument("--manifest_in", required=True)
    parser.add_argument("--rois_dir", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    clips_dir = Path(args.clips_dir)
    rois_dir = Path(args.rois_dir)
    rois_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.manifest_in) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    print(f"Processing {len(rows)} clips")
    
    print("Loading models...")
    det = init_detection_model("retinaface_resnet50", half=False, device="cuda:0")
    align = init_alignment_model("awing_fan", device="cuda:0")
    align.eval()
    video_process = VideoProcess(convert_gray=True)
    print("Loaded\n")
    
    out_rows = []
    t_start = time.time()
    failed = []
    
    for i, row in enumerate(rows):
        clip_path = clips_dir / row['path']
        clip_id = row['clip_id']
        if not clip_path.exists():
            print(f"  [{i+1}/{len(rows)}] {clip_id}: SKIP (not found)")
            failed.append(clip_id); continue
        t0 = time.time()
        try:
            rois, detected, total_frames = process_clip(clip_path, det, align, video_process)
        except Exception as e:
            print(f"  [{i+1}/{len(rows)}] {clip_id}: FAIL ({type(e).__name__}: {str(e)[:200]})")
            failed.append(clip_id); continue
        if rois is None:
            print(f"  [{i+1}/{len(rows)}] {clip_id}: FAIL (0 frames detected)")
            failed.append(clip_id); continue
        out_path = rois_dir / f"{clip_id}.mp4"
        write_grayscale_mp4(rois, out_path)
        elapsed = time.time() - t0
        out_rows.append({
            'clip_id': clip_id, 'path': out_path.name,
            'duration': row.get('dur', '?'),
            'detection_rate': f"{detected}/{total_frames}",
            'roi_frames': rois.shape[0], 'text': row['text'],
        })
        print(f"  [{i+1}/{len(rows)}] {clip_id}: {detected}/{total_frames} → {rois.shape}, {elapsed:.1f}s")
    
    if out_rows:
        with open(args.manifest_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader(); w.writerows(out_rows)
    
    total = time.time() - t_start
    print(f"\n=== DONE ===")
    print(f"Success: {len(out_rows)}/{len(rows)}")
    print(f"Failed: {len(failed)}")
    if failed: print(f"Failed clips: {failed[:10]}")
    print(f"Total: {total:.0f}s ({total/max(1,len(rows)):.1f}s/clip)")


if __name__ == "__main__":
    main()
