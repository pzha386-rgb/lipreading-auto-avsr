#!/bin/bash
# run_adapt_all.sh  -  train Auto-AVSR on 5/15/30/60/120/all minutes, then test each model
# Finished parts are skipped, so you can run it again after a stop.
#   cd /root/autodl-tmp/vsr_bench
#   nohup bash run_adapt_all.sh > /root/autodl-tmp/logs/adapt_all.log 2>&1 &

cd /root/autodl-tmp/vsr_bench
BASE=/root/autodl-tmp/checkpoints/vsr_trlrs2lrs3vox2avsp_base.pth
CONF=/root/autodl-tmp/day1/conf
DATASETS="tomscott_test bbc lrs3_300"

evaluate () {   # $1 = id, $2 = checkpoint
  for ds in $DATASETS; do
    if [ -f "$CONF/$1_${ds}_confA.csv" ]; then
      echo "skip eval $1 on $ds (done)"
    else
      echo ">>> eval $1 on $ds   $(date +%H:%M)"
      python -u eval_autoavsr_nbest.py --id "$1" --ckpt "$2" --dataset "$ds" | grep -E "final WER|saved"
    fi
  done
}

echo "===== baseline T1 (no adaptation)   $(date +%H:%M)"
evaluate T1 "$BASE"

for M in 5 15 30 60 120 0; do
  NAME=$([ "$M" = "0" ] && echo all || echo "$M")
  OUT=/root/autodl-tmp/adapt_v2/autoavsr_m$M
  echo
  echo "===== adapt $NAME min   $(date +%H:%M)"
  if [ -f "$OUT/best.pth" ] && grep -q '"history"' "$OUT/history.json" 2>/dev/null && [ -f "$OUT/done" ]; then
    echo "skip training (done)"
  else
    python -u adapt_autoavsr.py --minutes "$M" --out "$OUT" --freeze-layers 4 && touch "$OUT/done"
  fi
  [ -f "$OUT/best.pth" ] && evaluate "A$NAME" "$OUT/best.pth"
done

echo
echo "===== ALL DONE   $(date +%H:%M)"
python -u adapt_summary.py
