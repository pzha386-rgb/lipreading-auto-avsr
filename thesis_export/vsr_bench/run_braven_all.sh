#!/bin/bash
# run_braven_all.sh  -  evaluate B2 (BRAVEn low +ST +LM) on all test sets
# Finished parts are skipped.
#   cd /root/autodl-tmp/vsr_bench
#   nohup bash run_braven_all.sh > /root/autodl-tmp/logs/braven_all.log 2>&1 &

cd /root/autodl-tmp/vsr_bench
CK=/root/autodl-tmp/checkpoints/raven/vsr_prelrs3vox2avs_large_ftlrs3trainvalvox2avs_selftrain_braven.pth
LM=/root/autodl-tmp/checkpoints/raven/rnnlm.model.best
CONF=/root/autodl-tmp/day1/conf

for ds in tomscott_test bbc lrs3_300 tomscott; do
  OUT="$CONF/B2_${ds}_confA.csv"
  if [ -f "$OUT" ]; then
    echo "skip B2 on $ds (done)"
  else
    echo "===== B2 on $ds   $(date +%H:%M)"
    python -u eval_braven_nbest.py --id B2 --ckpt "$CK" --lm "$LM" --dataset "$ds" --gpu-decode
  fi
done

echo
echo "===== ALL DONE   $(date +%H:%M)"
echo "next: python calibrate_threshold.py --model B2"
