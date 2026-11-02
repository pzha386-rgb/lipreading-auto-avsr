# -*- coding: utf-8 -*-
"""
实时查看夜间任务进度

  python progress.py              # 每 60 秒刷新一次（Ctrl+C 退出）
  python progress.py -n 180       # 每 3 分钟刷新
  python progress.py --once       # 只看一次
"""
import argparse
import glob
import os
import re
import subprocess
import time

ROOT = "/root/autodl-tmp"
import sys
LOG = sys.argv[sys.argv.index("--log") + 1] if "--log" in sys.argv else f"{ROOT}/logs/night2.log"
RESULTS = f"{ROOT}/results_v2"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def running_tasks():
    """从进程列表找出正在跑的模型和数据集"""
    tasks = []
    for line in sh("ps -eww -o etimes,args | grep -v grep | grep 'eval_'").splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        secs = int(parts[0])
        args = " ".join(parts[1:])
        mid = re.search(r"--id (\S+)", args)
        ds = re.search(r"--dataset (\S+)", args)
        if mid and ds and (mid.group(1), ds.group(1)) not in [(t[0], t[1]) for t in tasks]:
            tasks.append((mid.group(1), ds.group(1), secs))
    return tasks


def task_progress(mid, ds):
    """从任务日志里读最后一个 [k/N]"""
    f = f"{ROOT}/logs/{mid}_{ds}.log"
    if not os.path.exists(f):
        return None
    last = None
    with open(f, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = re.match(r"\[(\d+)/(\d+)\] WER so far ([\d.]+)%", line)
            if m:
                last = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
    return last


def show():
    os.system("clear")
    print(f"========== 进度  {time.strftime('%H:%M:%S')} ==========\n")

    # 当前阶段
    log = open(LOG, encoding="utf-8", errors="ignore").read() if os.path.exists(LOG) else ""
    phases = re.findall(r"===== (\d\d:\d\d) (阶段[^=]*?) =====", log)
    if phases:
        t, p = phases[-1]
        print(f"当前阶段：{p.strip()}（{t} 开始）")
    if "全部完成" in log:
        print(">>> 夜间任务全部完成 <<<")
    done = log.count("<<<")
    failed = log.count("FAILED")
    print(f"本晚已完成任务：{done}    失败：{failed}\n")

    # 正在运行
    tasks = running_tasks()
    print(f"正在运行（{len(tasks)} 个）：")
    if not tasks:
        print("  （无）")
    for mid, ds, secs in tasks:
        p = task_progress(mid, ds)
        if p:
            k, n, wer = p
            bar = "#" * int(20 * k / n) + "-" * (20 - int(20 * k / n))
            eta = secs / k * (n - k) / 60 if k else 0
            print(f"  {mid:6s} {ds:12s} [{bar}] {k:4d}/{n:<4d}  WER {wer:5.1f}%  "
                  f"已用 {secs / 60:4.1f} 分  还需约 {eta:4.1f} 分")
        else:
            print(f"  {mid:6s} {ds:12s} 加载模型中…（已用 {secs / 60:.1f} 分）")

    # 最近完成的
    recent = [l for l in log.splitlines() if l.startswith("<<<")][-5:]
    if recent:
        print("\n最近完成：")
        for l in recent:
            print("  " + l[4:].replace(" on ", " | ").strip()[:90])

    # 结果文件数量
    n_res = {}
    for f in glob.glob(f"{RESULTS}/*.meta.json"):
        if "_limit" in f:
            continue
        ds = os.path.basename(f)[:-10].split("_", 1)[1]
        key = "鲁棒性" if ds.startswith("bbc_") else ds
        n_res[key] = n_res.get(key, 0) + 1
    print("\n已有结果：" + "   ".join(f"{k} {v}" for k, v in sorted(n_res.items())))

    # GPU 和磁盘
    gpu = sh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader").strip()
    disk = sh(f"df -h {ROOT} | tail -1").split()
    print(f"GPU：{gpu}")
    if len(disk) >= 5:
        print(f"磁盘：已用 {disk[2]} / {disk[1]}（{disk[4]}）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=60, help="刷新间隔（秒）")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    if a.once:
        show()
    else:
        try:
            while True:
                show()
                print(f"\n（每 {a.n} 秒刷新，Ctrl+C 退出）")
                time.sleep(a.n)
        except KeyboardInterrupt:
            pass
