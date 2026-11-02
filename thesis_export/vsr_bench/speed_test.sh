#!/bin/bash
# 速度测试：GPU 上只跑 1 个模型，逐个运行，保证计时公平（论文 RQ4 用）
# 用 BBC 前 21 段：第 1 段是预热（CUDA 初始化），统计时去掉，用后 20 段算平均
#
# 运行：nohup bash speed_test.sh > /root/autodl-tmp/logs/speed.log 2>&1 &
# 查看：python progress.py --log /root/autodl-tmp/logs/speed.log

cd "$(dirname "$0")"
C=/root/autodl-tmp/cache

# 1) 建一个 21 段的小数据集（指向 BBC 缓存，不复制文件）
mkdir -p $C/bbc_speed
python - << 'EOF'
import csv
rows = list(csv.DictReader(open("/root/autodl-tmp/cache/bbc/manifest.csv", encoding="utf-8")))[:21]
with open("/root/autodl-tmp/cache/bbc_speed/manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["utt_id", "text", "npy"])
    w.writeheader()
    for r in rows:
        w.writerow({**r, "npy": "../bbc/" + r["npy"]})
print("bbc_speed: 21 段")
EOF

# 2) 确认 GPU 空闲
if pgrep -f "eval_" > /dev/null; then
  echo "还有评测任务在跑，速度测试不公平，先停掉它们"; exit 1
fi

# 3) 逐个运行（--jobs 1）
ALL="T1 S0 S1 S6 S2 S2lm S4 S4lm S5 S5lm T3 T4 R1 R2 R3 B1 B2 B3 B4 B5 U1 U2 U3 U4 U1g U2g U3g U4g U4b10"
python -u run_queue.py --datasets bbc_speed --ids $ALL --jobs 1 --threads 4

# 4) 汇总：去掉第 1 段预热，计算每段耗时和实时率
python - << 'EOF'
import csv, json, glob, os
import numpy as np, pandas as pd
R = "/root/autodl-tmp/results_v2"
dur = {}
for r in csv.DictReader(open("/root/autodl-tmp/cache/bbc_speed/manifest.csv", encoding="utf-8")):
    dur[r["utt_id"]] = len(np.load("/root/autodl-tmp/cache/bbc_speed/" + r["npy"])) / 25.0   # 25 fps
rows = []
for f in sorted(glob.glob(f"{R}/*_bbc_speed.csv")):
    mid = os.path.basename(f).split("_bbc_speed")[0]
    df = pd.read_csv(f).iloc[1:]                      # 去掉预热的第 1 段
    meta = json.load(open(f.replace(".csv", ".meta.json")))
    video_sec = sum(dur[u] for u in df.utt_id)
    rows.append({"id": mid, "sec_per_clip": round(df.seconds.mean(), 3),
                 "RTF": round(df.seconds.sum() / video_sec, 3),     # <1 表示比实时快
                 "params_M": meta.get("params_M"), "peak_gpu_MB": meta.get("peak_gpu_MB")})
out = pd.DataFrame(rows).sort_values("RTF")
out.to_csv(f"{R}/summary_speed.csv", index=False)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
print("\nRTF = 处理时间 / 视频时长；RTF < 1 表示能实时运行")
EOF
