# -*- coding: utf-8 -*-
"""
adapt_progress.py  -  live view of run_adapt_all.sh (refresh every 15 s, Ctrl+C to exit)
  python /root/autodl-tmp/vsr_bench/adapt_progress.py
"""
import os
import subprocess
import time

import pandas as pd

LOG = "/root/autodl-tmp/logs/adapt_v2.log"
CONF = "/root/autodl-tmp/day1/conf"
IDS = [("T1", "no adapt"), ("A5", "5 min"), ("A15", "15 min"), ("A30", "30 min"),
       ("A60", "60 min"), ("A120", "120 min"), ("Aall", "all 257 min")]
SETS = [("tomscott_test", "TomScott test"), ("bbc", "BBC"), ("lrs3_300", "LRS3")]


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def wer(rid, ds):
    p = f"{CONF}/{rid}_{ds}_confA.csv"
    if not os.path.exists(p):
        return "   -  "
    d = pd.read_csv(p, usecols=["errors", "n_words"])
    return f"{100 * d['errors'].sum() / max(d['n_words'].sum(), 1):6.2f}"


while True:
    alive = subprocess.run(["pgrep", "-f", "^bash run_adapt_all"], capture_output=True).returncode == 0
    lines = open(LOG, encoding="utf-8", errors="ignore").read().split("\n") if os.path.exists(LOG) else []
    lines = [l for l in lines if l.strip()]
    stage = next((l for l in reversed(lines) if l.startswith("=====")), "not started")
    epochs = [l for l in lines[-60:] if l.startswith("epoch") or "early stop" in l or l.startswith("DONE")
              or l.startswith("train:")]
    since = lines[lines.index(stage):] if stage in lines else []
    ev = [l for l in since if l.startswith(">>> eval") or "final WER" in l]

    os.system("clear")
    print(time.strftime("%H:%M:%S"), "   run_adapt_all:", "RUNNING" if alive else "STOPPED")
    print("=" * 74)
    print("NOW:", stage.strip("= "))
    if epochs and ev == [] and "baseline" not in stage:
        print("  training:")
        for l in [l for l in since if l.startswith(("train:", "epoch", "early", "DONE"))][-8:]:
            print("   ", l)
    for l in ev[-4:]:
        print("   ", l)
    print("=" * 74)
    print(f"{'WER %':16s}" + "".join(f"{n:>16s}" for _, n in SETS))
    for rid, name in IDS:
        print(f"{rid:5s}{name:>11s}" + "".join(f"{wer(rid, ds):>16s}" for ds, _ in SETS))
    print("=" * 74)
    print("GPU :", sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"))
    print("DISK:", sh("df -h /root/autodl-tmp | awk 'NR==2{print $3\" used / \"$2\"  (\"$4\" free)\"}'"))
    print("\nlower WER = better.  Ctrl+C to exit (training keeps running)")
    time.sleep(15)
