# -*- coding: utf-8 -*-
"""
progress.py  -  live view of the whole data pipeline (refresh every 10 s, Ctrl+C to exit)

  python /root/autodl-tmp/tomscott/progress.py
"""
import glob
import os
import subprocess
import time

T = "/root/autodl-tmp/tomscott"
CACHE = "/root/autodl-tmp/cache"
LOGS = "/root/autodl-tmp/logs"


def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def last_line(path, n=3000):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - n))
        txt = f.read().decode("utf-8", "ignore")
    lines = [l.strip() for l in txt.replace("\r", "\n").split("\n") if l.strip() and "WARN" not in l]
    return lines[-1][:110] if lines else ""


def grep(path, key):
    if not os.path.exists(path):
        return ""
    for l in reversed(open(path, encoding="utf-8", errors="ignore").read().split("\n")):
        if key in l:
            return l.strip()
    return ""


def count_lines(path):
    return max(0, sum(1 for _ in open(path, encoding="utf-8")) - 1) if os.path.exists(path) else 0


def status(run, done):
    return "RUNNING" if run else ("DONE   " if done else "waiting")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


while True:
    excl = set()
    if os.path.exists(f"{T}/clips/exclude.txt"):
        excl = {l.strip() for l in open(f"{T}/clips/exclude.txt") if l.strip()}
    n_vid = len([p for p in glob.glob(f"{T}/tomscott_adapt/*.mp4")
                 if os.path.basename(p)[:-4] not in excl])
    clip_done = len(glob.glob(f"{T}/clips/*/manifest.csv"))
    keep_file = f"{T}/clips/keep_manifest.csv"
    n_keep = count_lines(keep_file)
    n_npy = len([e for e in os.scandir(f"{CACHE}/tomscott") if e.name.endswith(".npy")]) \
        if os.path.isdir(f"{CACHE}/tomscott") else 0

    # "^\\S*python" = only real python jobs, not the waiting bash command that mentions them
    r_clip = running(r"^\S*python\S* .*make_clips_all")
    r_filt = running(r"^\S*python\S* .*filter_clips")
    r_roi = running(r"^\S*python\S* .*cache_tomscott_rois")
    r_split = running(r"^\S*python\S* .*make_split")
    r_pipe = running(r"^bash -c while")

    os.system("clear")
    print(time.strftime("%H:%M:%S"), "  pipeline waiter:", "on" if r_pipe else "off")
    print("=" * 78)
    print(f"1. CLIPS   [{status(r_clip, clip_done >= n_vid)}]  {clip_done} / {n_vid} videos")
    if r_clip:
        print("   " + last_line(f"{T}/clips.log"))
    print(f"2. FILTER  [{status(r_filt, os.path.exists(keep_file) and not r_filt)}]", end="  ")
    if r_filt:
        print("\n   " + last_line(f"{T}/filter.log"))
    else:
        print(grep(f"{T}/filter.log", "TOTAL kept") or "")
    print(f"3. ROI     [{status(r_roi, os.path.exists(f'{CACHE}/tomscott/manifest.csv') and not r_roi)}]  "
          f"{n_npy} / {n_keep or '?'} clips" + (f"  ({100 * n_npy / n_keep:.0f}%)" if n_keep else ""))
    if r_roi:
        print("   " + last_line(f"{LOGS}/roi_tomscott.log"))
    print(f"4. SPLIT   [{status(r_split, os.path.exists(f'{CACHE}/tomscott_split.json'))}]")
    if os.path.exists(f"{LOGS}/split.log"):
        for l in open(f"{LOGS}/split.log").read().strip().split("\n")[-4:-1]:
            print("   " + l)
    print("=" * 78)
    print("GPU :", sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"))
    print("CPU :", sh("top -bn1 | awk '/Cpu\\(s\\)/{print 100-$8\" % used\"}'"))
    print("DISK:", sh("df -h /root/autodl-tmp | awk 'NR==2{print $3\" used / \"$2\"  (\"$4\" free)\"}'"))
    print("\nCtrl+C to exit (the jobs keep running)")
    time.sleep(10)
