"""
make_clips.py - Parse SRT, merge adjacent lines into 3-5 second clips,
                cut video into clips at 25fps grayscale,
                generate manifest CSV.

Usage:
    py make_clips.py tom_scott.mp4 tom_scott.en.srt clips/ manifest.csv
"""
import sys
import re
import os
import csv
import subprocess
from pathlib import Path

# ---------------- SRT Parser ----------------

def time_to_seconds(t):
    """'00:01:23,450' -> 83.45"""
    h, m, rest = t.split(':')
    s, ms = rest.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def parse_srt(path):
    """Return list of dicts: {start, end, text}"""
    with open(path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    entries = []
    for block in blocks:
        lines = [l for l in block.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            continue
        # Find time line
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
        except ValueError:
            continue
        text = ' '.join(lines[time_idx + 1:]).strip()
        if not text:
            continue
        entries.append({'start': start, 'end': end, 'text': text})
    return entries

# ---------------- Merge into 3-5 second clips ----------------

def merge_clips(entries, min_dur=3.0, max_dur=5.0, hard_max=6.0):
    """
    Merge adjacent SRT entries so each merged clip is ~3-5 seconds.
    Priority: meet min_dur first (allow up to hard_max), prefer staying under max_dur.
    """
    clips = []
    i = 0
    n = len(entries)
    while i < n:
        start = entries[i]['start']
        end = entries[i]['end']
        text_parts = [entries[i]['text']]
        j = i + 1
        while j < n:
            cur_dur = end - start
            new_dur = entries[j]['end'] - start
            if cur_dur >= min_dur:
                # Already long enough — only extend if it stays under max_dur
                if new_dur > max_dur:
                    break
            else:
                # Must extend to reach min_dur, but cap at hard_max
                if new_dur > hard_max:
                    break
            end = entries[j]['end']
            text_parts.append(entries[j]['text'])
            j += 1
            if (end - start) >= max_dur:
                break

        text = ' '.join(text_parts)
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove music notes / sound markers in [...]
        text = re.sub(r'\[[^\]]*\]', '', text).strip()
        # Remove parenthetical (...)
        text = re.sub(r'\([^)]*\)', '', text).strip()
        # Final cleanup of multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            i = j
            continue
        clips.append({'start': start, 'end': end, 'text': text, 'dur': end - start})
        i = j

    # Tail-fix: if the last clip is too short (< min_dur) and we have at least 2 clips,
    # merge it into the previous one (allow slight overflow, capped at hard_max).
    if len(clips) >= 2 and clips[-1]['dur'] < min_dur:
        merged_dur = clips[-1]['end'] - clips[-2]['start']
        if merged_dur <= hard_max + 1.5:  # be a bit lenient on the very last one
            tail = clips.pop()
            clips[-1]['end'] = tail['end']
            clips[-1]['text'] = clips[-1]['text'] + ' ' + tail['text']
            clips[-1]['dur'] = clips[-1]['end'] - clips[-1]['start']

    return clips

# ---------------- ffmpeg cutter ----------------

def cut_clip(input_video, start, end, output_path):
    """
    Cut clip with re-encoding to 25fps grayscale 96x96 (LRS3 spec)?
    No - we cut to RGB at original resolution + 25fps.
    The mouth ROI cropping happens later in BitaHub.
    Here we only:
      - extract the time range
      - downsample 50fps -> 25fps
      - keep RGB (face detection later needs color or grayscale, both ok)
      - keep original resolution
    """
    duration = end - start
    cmd = [
        'ffmpeg', '-y',
        '-loglevel', 'error',
        '-ss', f'{start:.3f}',
        '-i', input_video,
        '-t', f'{duration:.3f}',
        '-r', '25',                # downsample to 25fps
        '-an',                     # drop audio (model is visual-only)
        '-c:v', 'libx264',
        '-crf', '18',              # near-lossless
        '-pix_fmt', 'yuv420p',
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg failed for {output_path}: {result.stderr}", file=sys.stderr)
        return False
    return True

# ---------------- Main ----------------

def main():
    if len(sys.argv) != 5:
        print("Usage: py make_clips.py <video.mp4> <subs.srt> <out_dir> <manifest.csv>")
        sys.exit(1)

    video_path = sys.argv[1]
    srt_path = sys.argv[2]
    out_dir = Path(sys.argv[3])
    manifest_path = sys.argv[4]

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading SRT: {srt_path}")
    entries = parse_srt(srt_path)
    print(f"  {len(entries)} raw subtitle entries")

    print(f"Merging into 3-5 second clips...")
    clips = merge_clips(entries, min_dur=3.0, max_dur=5.0)
    print(f"  {len(clips)} merged clips")

    # Print stats
    durs = [c['dur'] for c in clips]
    if durs:
        print(f"  duration: min={min(durs):.2f}s, max={max(durs):.2f}s, "
              f"avg={sum(durs)/len(durs):.2f}s, total={sum(durs):.1f}s")

    # Cut clips
    print(f"\nCutting clips with ffmpeg (25fps, no audio)...")
    manifest_rows = []
    success = 0
    for idx, clip in enumerate(clips):
        clip_id = f"clip_{idx:04d}"
        out_path = out_dir / f"{clip_id}.mp4"
        ok = cut_clip(video_path, clip['start'], clip['end'], out_path)
        if ok:
            success += 1
            manifest_rows.append({
                'clip_id': clip_id,
                'path': str(out_path.name),  # just filename, container will resolve
                'start': f"{clip['start']:.3f}",
                'end': f"{clip['end']:.3f}",
                'dur': f"{clip['dur']:.3f}",
                'text': clip['text'],
            })
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(clips)} done")

    print(f"\n{success}/{len(clips)} clips cut successfully")

    # Write manifest
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['clip_id', 'path', 'start', 'end', 'dur', 'text'])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest written to: {manifest_path}")
    print(f"Clips dir: {out_dir.resolve()}")

if __name__ == '__main__':
    main()
