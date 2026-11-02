#!/bin/bash
# 第 1 天夜间任务：两条队列同时跑
#   队列 A：Auto-AVSR / VSR-ML / RAVEn / BRAVEn（每个占 GPU 约 3 GB，同时 4 个）
#   队列 B：USR 2.0（每个占 GPU 13 GB 以上，一次只跑 1 个）
# 每条队列按阶段执行：BBC 补齐 -> LRS3 300 段 -> 鲁棒性 8 种条件 -> LRS3 完整 1321 段
# 已完成的自动跳过，权重还没下载完的自动跳过（下一阶段会再检查一次）
#
# 运行：nohup bash night1.sh > /root/autodl-tmp/logs/night1.log 2>&1 &
# 查看：grep "阶段\|<<<" /root/autodl-tmp/logs/night1_A.log | tail -20

cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/root/autodl-tmp/logs

A_ALL="T1 T3 T4 S0 S1 S6 S2 S2lm S4 S4lm S5 S5lm R1 R2 R3 B1 B2 B3 B4 B5"
B_ALL="U1 U2 U3 U4 U1g U2g U3g U4g U4b10"
ROBUST="bbc_blur2 bbc_blur4 bbc_low50 bbc_low25 bbc_dark05 bbc_dark03 bbc_drop25 bbc_drop50"
A_ROBUST="T1 S4 T3 T4 B4"          # 监督（自动标签）、纯监督、BRAVEn 低/高资源、RAVEn
B_ROBUST="U4 U3"                    # USR 2.0（U4 没下载完就用 U3）

# 等之前手动启动的 BBC 任务结束，避免重复运行同一个模型
while pgrep -f "run_queue.py --datasets bbc" > /dev/null; do
  echo "$(date +%H:%M) 等待之前的 BBC 任务结束..."; sleep 60
done

queue_a() {
  echo "===== $(date +%H:%M) 阶段 1：BBC 补齐 ====="
  python -u run_queue.py --datasets bbc --ids $A_ALL --jobs 4 --threads 2
  echo "===== $(date +%H:%M) 阶段 2：LRS3 300 段 ====="
  python -u run_queue.py --datasets lrs3_300 --ids $A_ALL --jobs 4 --threads 2
  echo "===== $(date +%H:%M) 阶段 3：鲁棒性 ====="
  python -u run_queue.py --datasets $ROBUST --ids $A_ROBUST --jobs 4 --threads 2
  echo "===== $(date +%H:%M) 阶段 4：BBC 和 LRS3 300 再补一次（之前没下载完的 S0 S1）====="
  python -u run_queue.py --datasets bbc lrs3_300 --ids $A_ALL --jobs 4 --threads 2
  echo "===== $(date +%H:%M) 阶段 5：LRS3 完整 1321 段 ====="
  python -u run_queue.py --datasets lrs3 --ids $A_ALL --jobs 4 --threads 2
  echo "===== $(date +%H:%M) 队列 A 全部完成 ====="
}

queue_b() {
  echo "===== $(date +%H:%M) 阶段 1：BBC ====="
  python -u run_queue.py --datasets bbc --ids $B_ALL --jobs 1 --threads 2
  echo "===== $(date +%H:%M) 阶段 2：LRS3 300 段 ====="
  python -u run_queue.py --datasets lrs3_300 --ids $B_ALL --jobs 1 --threads 2
  echo "===== $(date +%H:%M) 阶段 3：鲁棒性 ====="
  python -u run_queue.py --datasets $ROBUST --ids $B_ROBUST --jobs 1 --threads 2
  echo "===== $(date +%H:%M) 阶段 4：再补一次（之前没下载完的 U4）====="
  python -u run_queue.py --datasets bbc lrs3_300 $ROBUST --ids U4 U4g U4b10 --jobs 1 --threads 2
  echo "===== $(date +%H:%M) 阶段 5：LRS3 完整 1321 段 ====="
  python -u run_queue.py --datasets lrs3 --ids $B_ALL --jobs 1 --threads 2
  echo "===== $(date +%H:%M) 队列 B 全部完成 ====="
}

queue_a > $LOG/night1_A.log 2>&1 &
queue_b > $LOG/night1_B.log 2>&1 &
wait
echo "===== $(date +%H:%M) 全部完成 ====="
python run_queue.py --summary
