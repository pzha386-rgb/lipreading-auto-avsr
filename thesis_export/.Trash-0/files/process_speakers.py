"""
process_speakers.py - Multi-speaker SRT→clips→face-filter pipeline
====================================================================
Scans vsr_speakers/train/ and vsr_speakers/test/, for each speaker/video:
  1. Parse SRT → merge into 3-5s clips
  2. Cut clips with ffmpeg (25fps, no audio)
  3. Face-filter: keep clips where a large frontal face appears in most frames
  4. Write per-speaker manifests + global summary

Output structure:
  OUTPUT_DIR/
    train/
      mkbhd/
        VIDEO_ID/clip_0000.mp4, clip_0001.mp4, ...
      ali_abdaal/
        ...
    test/
      fireship/
        ...
    manifests/
      train_mkbhd.csv
      train_ali_abdaal.csv
      test_fireship.csv
      ...
      train_all.csv       ← all train clips
      test_all.csv        ← all test clips
    summary.txt

Usage (on AutoDL GPU server):
    python process_speakers.py

Requirements: ffmpeg, opencv-python (already installed)
"""

import os
import re
import csv
import sys
import subprocess
import glob
from pathlib import Path
from multiprocessing import Pool
import cv2
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_DIR = "/root/autodl-tmp/vsr_speakers"
OUTPUT_DIR = "/root/autodl-tmp/speaker_clips"

# Clip merging
MIN_DUR = 3.0
MAX_DUR = 5.0
HARD_MAX = 6.0

# Face filter
SAMPLE_FRACS = (0.2, 0.5, 0.8)    # sample 3 frames per clip
MIN_FACE_WIDTH = 50                 # pixels, face must be this wide
MAX_YAW = 0.35                      # frontal threshold
FACE_AREA_RATIO = 0.02              # face must be >2% of frame
KEEP_THRESHOLD = 2                  # need >=2 good frames out of 3

# Workers for face filtering
NUM_WORKERS = min(8, max(2, (os.cpu_count() or 4) - 2))

# DNN face detector (OpenCV built-in, no download needed)
DNN_PROTO = None  # will use Haar cascade as fallback
DNN_MODEL = None


# ============================================================
# SRT PARSING (from your make_clips.py)
# ============================================================

def time_to_seconds(t):
    h, m, rest = t.split(':')
    s, ms = rest.split(',')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def parse_srt(path):
    with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        content = f.read()
    blocks = re.split(r'\n\s*\n', content.strip())
    entries = []
    for block in blocks:
        lines = [l for l in block.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            continue
        time_idx = None
        for i, line in enumerate(lines):
            if '-->' in line:
                time_idx = i
                break
        if time_idx is None:
            continue
        try:
            start_str, end_str = lines[time_idx].split('-->')
            start = time_to_seconds(start_str.strip())
            end = time_to_seconds(end_str.strip())
        except (ValueError, IndexError):
            continue
        text = ' '.join(lines[time_idx+1:]).strip()
        if not text:
            continue
        entries.append({'start': start, 'end': end, 'text': text})
    return entries

def merge_clips(entries, min_dur=MIN_DUR, max_dur=MAX_DUR, hard_max=HARD_MAX):
    clips = []
    i, n = 0, len(entries)
    while i < n:
        start = entries[i]['start']
        end = entries[i]['end']
        text_parts = [entries[i]['text']]
        j = i + 1
        while j < n:
            cur_dur = end - start
            new_dur = entries[j]['end'] - start
            if cur_dur >= min_dur:
                if new_dur > max_dur:
                    break
            else:
                if new_dur > hard_max:
                    break
            end = entries[j]['end']
            text_parts.append(entries[j]['text'])
            j += 1
            if (end - start) >= max_dur:
                break
        text = ' '.join(text_parts)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\[[^\]]*\]', '', text).strip()
        text = re.sub(r'\([^)]*\)', '', text).strip()
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            i = j
            continue
        clips.append({'start': start, 'end': end, 'text': text, 'dur': end-start})
        i = j
    if len(clips) >= 2 and clips[-1]['dur'] < min_dur:
        merged_dur = clips[-1]['end'] - clips[-2]['start']
        if merged_dur <= hard_max + 1.5:
            tail = clips.pop()
            clips[-1]['end'] = tail['end']
            clips[-1]['text'] += ' ' + tail['text']
            clips[-1]['dur'] = clips[-1]['end'] - clips[-1]['start']
    return clips


# ============================================================
# FFMPEG CLIP CUTTING
# ============================================================

def cut_clip(input_video, start, end, output_path):
    duration = end - start
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-ss', f'{start:.3f}', '-i', input_video,
        '-t', f'{duration:.3f}',
        '-r', '25', '-an',
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# ============================================================
# FACE FILTER (simplified: no identity check, just face presence)
# ============================================================

_cascade = None

def init_detector():
    global _cascade
    cv2.setNumThreads(1)
    _cascade = None  # 
        ""  # removed
    )
    # Try DNN if model files exist
    proto = "/root/autodl-tmp/tomscott/models/face_detection_yunet_2023mar.onnx"
    if os.path.exists(proto):
        # Use YuNet detector (much better than Haar)
        global _use_yunet
        _use_yunet = True
    else:
        _use_yunet = False

_use_yunet = False
_yunet = None

def init_worker():
    global _cascade, _yunet, _use_yunet
    cv2.setNumThreads(1)
    _cascade = None  # 
        ""  # removed
    )
    yunet_path = "/root/autodl-tmp/tomscott/models/face_detection_yunet_2023mar.onnx"
    if os.path.exists(yunet_path):
        _yunet = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.7, 0.3, 5000)
        _use_yunet = True
    else:
        _use_yunet = False


def detect_faces_yunet(frame):
    h, w = frame.shape[:2]
    s = 0.5 if w > 640 else 1.0
    small = cv2.resize(frame, (int(w*s), int(h*s))) if s != 1.0 else frame
    _yunet.setInputSize((small.shape[1], small.shape[0]))
    _, faces = _yunet.detect(small)
    if faces is None:
        return []
    faces = faces.copy()
    faces[:, :14] /= s
    return faces


def detect_faces_haar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    return faces


def check_frame_quality(frame):
    """Check if frame has a good frontal face. Returns 'ok' or reason."""
    h, w = frame.shape[:2]
    frame_area = h * w

    if _use_yunet and _yunet is not None:
        faces = detect_faces_yunet(frame)
        if len(faces) == 0:
            return "no_face"
        if len(faces) > 3:
            return "multi_face"
        # Largest face
        best = max(faces, key=lambda f: f[2] * f[3])
        face_w = float(best[2])
        face_area = float(best[2]) * float(best[3])
        if face_area / frame_area < FACE_AREA_RATIO:
            return "small_face"
        if face_w < MIN_FACE_WIDTH:
            return "small_face"
        # Yaw check using landmarks
        if best.shape[0] >= 10:
            rex, lex, nx = float(best[4]), float(best[6]), float(best[8])
            eye_d = abs(lex - rex) + 1e-6
            yaw = abs((nx - (rex + lex) / 2) / eye_d)
            if yaw >= MAX_YAW:
                return "side_face"
        return "ok"
    else:
        # Haar fallback
        faces = detect_faces_haar(frame)
        if len(faces) == 0:
            return "no_face"
        if len(faces) > 3:
            return "multi_face"
        best_area = max(fw * fh for (_, _, fw, fh) in faces)
        if best_area / frame_area < FACE_AREA_RATIO:
            return "small_face"
        return "ok"


def check_clip_task(task):
    """Check a single clip. task = (clip_path, clip_info_dict)"""
    clip_path, info = task
    try:
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return {**info, "keep": False, "reason": "unreadable", "n_ok": 0}
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 3:
            cap.release()
            return {**info, "keep": False, "reason": "too_short", "n_ok": 0}

        results = []
        for frac in SAMPLE_FRACS:
            idx = min(int(total * frac), total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                results.append("unreadable")
                continue
            results.append(check_frame_quality(frame))
        cap.release()

        n_ok = sum(1 for r in results if r == "ok")
        keep = n_ok >= KEEP_THRESHOLD
        if keep:
            reason = "ok"
        else:
            from collections import Counter
            bad = [r for r in results if r != "ok"]
            reason = Counter(bad).most_common(1)[0][0] if bad else "unknown"

        return {**info, "keep": keep, "reason": reason, "n_ok": n_ok}
    except Exception as e:
        return {**info, "keep": False, "reason": f"error:{e}", "n_ok": 0}


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_video(video_path, srt_path, speaker_id, video_id, split, clip_out_dir):
    """Process one video: SRT → clips → return list of clip info dicts."""
    os.makedirs(clip_out_dir, exist_ok=True)

    # Check if already processed
    existing = [f for f in os.listdir(clip_out_dir) if f.endswith('.mp4')]
    manifest_file = os.path.join(clip_out_dir, "manifest.csv")
    if existing and os.path.exists(manifest_file):
        # Already done, read manifest
        rows = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows

    entries = parse_srt(srt_path)
    if not entries:
        return []

    clips = merge_clips(entries)
    if not clips:
        return []

    rows = []
    for idx, clip in enumerate(clips):
        clip_id = f"clip_{idx:04d}"
        out_path = os.path.join(clip_out_dir, f"{clip_id}.mp4")
        ok = cut_clip(video_path, clip['start'], clip['end'], out_path)
        if ok:
            rows.append({
                'clip_id': clip_id,
                'video_id': video_id,
                'speaker_id': speaker_id,
                'split': split,
                'path': os.path.join(split, speaker_id, video_id, f"{clip_id}.mp4"),
                'start': f"{clip['start']:.3f}",
                'end': f"{clip['end']:.3f}",
                'dur': f"{clip['dur']:.3f}",
                'text': clip['text'],
            })

    # Save per-video manifest
    if rows:
        with open(manifest_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return rows


def main():
    print("=" * 60)
    print("Multi-Speaker Processing Pipeline")
    print("=" * 60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "manifests"), exist_ok=True)

    # ---- STEP 1: Clip extraction ----
    print("STEP 1: Extracting clips from SRT...")
    all_rows = []
    speaker_stats = {}

    for split in ["train", "test"]:
        split_dir = os.path.join(INPUT_DIR, split)
        if not os.path.exists(split_dir):
            continue

        for speaker_id in sorted(os.listdir(split_dir)):
            spk_dir = os.path.join(split_dir, speaker_id)
            if not os.path.isdir(spk_dir):
                continue

            # Find all video+srt pairs
            mp4s = sorted([f for f in os.listdir(spk_dir) if f.endswith('.mp4')])
            if not mp4s:
                continue

            print(f"\n  {split}/{speaker_id}: {len(mp4s)} videos")
            spk_clips = 0
            spk_dur = 0.0

            for mp4 in mp4s:
                video_id = mp4.replace('.mp4', '')
                video_path = os.path.join(spk_dir, mp4)

                # Find matching SRT
                srt_path = None
                for ext in ['.en.srt', '.srt']:
                    candidate = os.path.join(spk_dir, video_id + ext)
                    if os.path.exists(candidate):
                        srt_path = candidate
                        break
                if not srt_path:
                    # Try any srt with video_id in name
                    for f in os.listdir(spk_dir):
                        if video_id in f and f.endswith('.srt'):
                            srt_path = os.path.join(spk_dir, f)
                            break
                if not srt_path:
                    print(f"    {video_id}: no SRT found, skipping")
                    continue

                clip_out_dir = os.path.join(OUTPUT_DIR, split, speaker_id, video_id)
                rows = process_video(video_path, srt_path, speaker_id, video_id, split, clip_out_dir)
                all_rows.extend(rows)
                n = len(rows)
                d = sum(float(r['dur']) for r in rows)
                spk_clips += n
                spk_dur += d
                print(f"    {video_id}: {n} clips, {d/60:.1f} min")

            speaker_stats[(split, speaker_id)] = {
                'clips': spk_clips, 'dur_min': spk_dur / 60
            }
            print(f"    TOTAL: {spk_clips} clips, {spk_dur/60:.1f} min")

    print(f"\nSTEP 1 DONE: {len(all_rows)} clips total")

    # ---- STEP 2: Face filtering ----
    print(f"\nSTEP 2: Face filtering ({NUM_WORKERS} workers)...")
    print("  (This checks each clip for a clear frontal face)")

    tasks = []
    for row in all_rows:
        clip_path = os.path.join(OUTPUT_DIR, row['path'])
        tasks.append((clip_path, row))

    from tqdm import tqdm
    with Pool(NUM_WORKERS, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(check_clip_task, tasks, chunksize=8),
            total=len(tasks), ncols=80, desc="  Filtering"
        ))

    # ---- STEP 3: Save manifests ----
    print(f"\nSTEP 3: Saving manifests...")

    kept = [r for r in results if r['keep']]
    rejected = [r for r in results if not r['keep']]

    print(f"  Kept: {len(kept)} / {len(results)} clips")
    print(f"  Rejected: {len(rejected)} clips")

    # Rejection reasons
    from collections import Counter
    reasons = Counter(r['reason'] for r in rejected)
    for reason, count in reasons.most_common():
        print(f"    {reason}: {count}")

    # Per-speaker manifests
    manifest_dir = os.path.join(OUTPUT_DIR, "manifests")
    fields = ['clip_id', 'video_id', 'speaker_id', 'split', 'path', 'start', 'end', 'dur', 'text']

    train_all = []
    test_all = []

    for (split, speaker_id), stats in sorted(speaker_stats.items()):
        spk_kept = [r for r in kept if r['speaker_id'] == speaker_id and r['split'] == split]
        if not spk_kept:
            continue

        manifest_path = os.path.join(manifest_dir, f"{split}_{speaker_id}.csv")
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(spk_kept)

        if split == "train":
            train_all.extend(spk_kept)
        else:
            test_all.extend(spk_kept)

    # Global manifests
    for name, data in [("train_all", train_all), ("test_all", test_all)]:
        path = os.path.join(manifest_dir, f"{name}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)

    # ---- SUMMARY ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Speaker':<22} {'Split':<7} {'Raw':<6} {'Kept':<6} {'Keep%':<7} {'Minutes'}")
    print("-" * 65)

    for (split, speaker_id), stats in sorted(speaker_stats.items()):
        spk_kept = [r for r in kept if r['speaker_id'] == speaker_id and r['split'] == split]
        spk_all = [r for r in results if r['speaker_id'] == speaker_id and r['split'] == split]
        keep_pct = 100 * len(spk_kept) / len(spk_all) if spk_all else 0
        kept_min = sum(float(r['dur']) for r in spk_kept) / 60
        print(f"  {speaker_id:<20} {split:<7} {len(spk_all):<6} {len(spk_kept):<6} {keep_pct:<7.1f} {kept_min:.1f}")

    train_min = sum(float(r['dur']) for r in train_all) / 60
    test_min = sum(float(r['dur']) for r in test_all) / 60

    print("-" * 65)
    print(f"  TRAIN: {len(train_all)} clips, {train_min:.1f} min")
    print(f"  TEST:  {len(test_all)} clips, {test_min:.1f} min")
    print(f"  TOTAL: {len(kept)} clips, {train_min + test_min:.1f} min")

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Train: {len(train_all)} clips, {train_min:.1f} min\n")
        f.write(f"Test:  {len(test_all)} clips, {test_min:.1f} min\n")
        f.write(f"Total: {len(kept)} clips, {train_min + test_min:.1f} min\n\n")
        for (split, speaker_id), _ in sorted(speaker_stats.items()):
            spk_kept = [r for r in kept if r['speaker_id'] == speaker_id and r['split'] == split]
            kept_min = sum(float(r['dur']) for r in spk_kept) / 60
            f.write(f"{split}/{speaker_id}: {len(spk_kept)} clips, {kept_min:.1f} min\n")

    print(f"\nManifests: {manifest_dir}/")
    print(f"Clips:     {OUTPUT_DIR}/train/ and {OUTPUT_DIR}/test/")
    print(f"\nNext step: ROI extraction (mouth crop 96x96)")


if __name__ == "__main__":
    main()
