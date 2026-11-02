"""
extract_roi.py — Multi-speaker mouth ROI extraction
Uses the EXACT same pipeline that worked for Tom Scott BBC test:
  facexlib RetinaFace + manual AwingFAN decode + 98→68 + ibug video_process
"""
import sys, time, csv, os, warnings
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision

warnings.filterwarnings("ignore")

# Monkey patch for numpy compatibility
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool

PREP_DIR = "/root/autodl-tmp/code/auto_avsr/preparation"
sys.path.insert(0, PREP_DIR)
os.chdir(PREP_DIR)

from facexlib.detection import init_detection_model
from facexlib.alignment import init_alignment_model
from facexlib.alignment.convert_98_to_68_landmarks import landmark_98_to_68
from detectors.retinaface.video_process import VideoProcess
from transforms import TextTransform

CLIPS_DIR = "/root/autodl-tmp/speaker_clips"
OUTPUT_DIR = "/root/autodl-tmp/speaker_roi"
DATASET_NAME = "multi_speaker"
SEG_DURATION = 16
WEIGHTS_DIR = "/root/autodl-tmp/facexlib_weights"


def decode_landmark(align, frame_rgb, bbox, margin=0.3):
    """Manual landmark extraction — the EXACT method that worked before."""
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


def process_clip(clip_path, det, align, vp):
    """Process one clip -> 96x96 grayscale ROI or None."""
    video = torchvision.io.read_video(str(clip_path), pts_unit="sec")[0].numpy()
    T = video.shape[0]
    landmarks = []
    detected = 0
    for t in range(T):
        if t % 3 != 0 and t < T - 1:
            landmarks.append(None)
            continue
        frame_rgb = video[t]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            bboxes = det.detect_faces(frame_bgr, 0.5)
        if len(bboxes) == 0:
            landmarks.append(None)
            continue
        best = max(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        bbox = [float(best[0]), float(best[1]), float(best[2]), float(best[3])]
        lm98 = decode_landmark(align, frame_rgb, bbox)
        if lm98 is None:
            landmarks.append(None)
            continue
        lm68 = landmark_98_to_68(lm98)
        landmarks.append(lm68)
        detected += 1
    if detected == 0:
        return None, 0, T
    rois = vp(video, landmarks)
    return rois, detected, T


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
    os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

    print("Loading facexlib models...")
    det = init_detection_model("retinaface_resnet50", half=True, device="cuda:0",
                                model_rootpath=WEIGHTS_DIR)
    align = init_alignment_model("awing_fan", device="cuda:0",
                                  model_rootpath=WEIGHTS_DIR)
    align.eval()
    vp = VideoProcess(convert_gray=True)
    text_transform = TextTransform()
    print("Models ready.\n")

    for split in ["test"]:
        manifest_path = os.path.join(CLIPS_DIR, "manifests", f"{split}_all.csv")
        if not os.path.exists(manifest_path):
            print(f"Manifest not found: {manifest_path}")
            continue
        with open(manifest_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        print(f"\n{'='*60}")
        print(f"ROI Extraction: {split} ({len(rows)} clips)")
        print(f"{'='*60}")

        label_path = os.path.join(OUTPUT_DIR, "labels",
            f"{DATASET_NAME}_{split}_transcript_lengths_seg{SEG_DURATION}s.csv")
        label_file = open(label_path, 'w', encoding='utf-8')
        success, fail = 0, 0
        speaker_stats = {}
        t_start = time.time()

        for i, row in enumerate(rows):
            clip_rel = row['path']
            clip_abs = os.path.join(CLIPS_DIR, clip_rel)
            speaker_id = row['speaker_id']
            text = row['text']
            clip_id = row['clip_id']
            video_id = row['video_id']

            dst_rel = os.path.join(
                f"{DATASET_NAME}_video_seg{SEG_DURATION}s",
                split, speaker_id, video_id, f"{clip_id}.mp4")
            dst_abs = os.path.join(OUTPUT_DIR, DATASET_NAME, dst_rel)

            # Skip if done
            if os.path.exists(dst_abs) and os.path.getsize(dst_abs) > 100:
                token_id_str = " ".join(
                    map(str, [_.item() for _ in text_transform.tokenize(text)]))
                if token_id_str:
                    vlen = int(float(row['dur']) * 25)
                    label_file.write(f"{DATASET_NAME},{dst_rel},{vlen},{token_id_str}\n")
                    success += 1
                    speaker_stats[speaker_id] = speaker_stats.get(speaker_id, 0) + 1
                continue

            try:
                rois, detected, total = process_clip(clip_abs, det, align, vp)
            except Exception as e:
                fail += 1
                if (i+1) % 100 == 0:
                    print(f"  [{i+1}/{len(rows)}] FAIL: {e}")
                continue

            if rois is None:
                fail += 1
                continue

            os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
            write_grayscale_mp4(rois, dst_abs)
            token_id_str = " ".join(
                map(str, [_.item() for _ in text_transform.tokenize(text)]))
            if token_id_str:
                label_file.write(f"{DATASET_NAME},{dst_rel},{rois.shape[0]},{token_id_str}\n")
                success += 1
                speaker_stats[speaker_id] = speaker_stats.get(speaker_id, 0) + 1

            if (i+1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (i+1) / elapsed
                eta = (len(rows) - i - 1) / rate
                print(f"  [{i+1}/{len(rows)}] {success} ok, {fail} fail | "
                      f"{rate:.1f} clips/s | ETA {eta/60:.0f} min")

        label_file.close()
        elapsed = time.time() - t_start
        print(f"\n  {split}: {success} success, {fail} failed, {elapsed/60:.1f} min")
        for sid, cnt in sorted(speaker_stats.items()):
            print(f"    {sid:<22} {cnt} clips")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
