#!/bin/bash
# 夜间任务（修正版）：两类模型先后执行，避免 GPU 显存不足
#   第一部分：Auto-AVSR / VSR-ML / RAVEn / BRAVEn，同时 5 个（每个约 3-5 GB，共约 24 GB）
#   第二部分：USR 2.0，独占 GPU，一次 1 个（Large 约 15 GB，Huge 更多）
#   第三部分：完整 LRS3（先第一部分的模型，再 USR 2.0 的 U3、U4）
# 已完成的自动跳过；失败的会在后面的阶段自动补跑一次
#
# 运行：nohup bash night2.sh > /root/autodl-tmp/logs/night2.log 2>&1 &
# 查看：grep "阶段\|<<<" /root/autodl-tmp/logs/night2.log | tail -20

cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

A_ALL="T1 T3 T4 S0 S1 S6 S2 S2lm S4 S4lm S5 S5lm R1 R2 R3 B1 B2 B3 B4 B5"
B_ALL="U1 U2 U3 U4 U1g U2g U3g U4g U4b10"
ROBUST="bbc_blur2 bbc_blur4 bbc_low50 bbc_low25 bbc_dark05 bbc_dark03 bbc_drop25 bbc_drop50"
A_ROBUST="T1 S4 T3 T4 B4"
B_ROBUST="U4"

run() { echo "===== $(date +%H:%M) $1 ====="; shift; python -u run_queue.py "$@"; }

# ---- 第一部分：A 类模型 ----
run "阶段 1：A 类 BBC + LRS3 300"   --datasets bbc lrs3_300 --ids $A_ALL --jobs 5 --threads 2
run "阶段 2：A 类鲁棒性"            --datasets $ROBUST --ids $A_ROBUST --jobs 5 --threads 2
run "阶段 3a：A 类补跑失败的"       --datasets bbc lrs3_300 --ids $A_ALL --jobs 3 --threads 2
run "阶段 3b：A 类鲁棒性补跑"        --datasets $ROBUST --ids $A_ROBUST --jobs 3 --threads 2

# ---- 第二部分：USR 2.0（独占 GPU）----
run "阶段 4：USR 2.0 BBC + LRS3 300" --datasets bbc lrs3_300 --ids $B_ALL --jobs 1 --threads 4
run "阶段 5：USR 2.0 鲁棒性"         --datasets $ROBUST --ids $B_ROBUST --jobs 1 --threads 4
run "阶段 6a：USR 2.0 补跑失败的"    --datasets bbc lrs3_300 --ids $B_ALL --jobs 1 --threads 4
run "阶段 6b：USR 2.0 鲁棒性补跑"    --datasets $ROBUST --ids $B_ROBUST --jobs 1 --threads 4

# ---- 第三部分：完整 LRS3 ----
run "阶段 7：A 类完整 LRS3"          --datasets lrs3 --ids $A_ALL --jobs 5 --threads 2
run "阶段 8：USR 2.0 完整 LRS3"      --datasets lrs3 --ids U4 U3 --jobs 1 --threads 4

echo "===== $(date +%H:%M) 全部完成 ====="
python run_queue.py --summary
