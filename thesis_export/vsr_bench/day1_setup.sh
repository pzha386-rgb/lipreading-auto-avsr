#!/bin/bash
# 第 1 天 12:00 运行：装包 + 检查环境 + 找到需要填进 models.csv 的文件
# 运行：bash day1_setup.sh 2>&1 | tee /root/autodl-tmp/logs/setup.log

R=/root/autodl-tmp
mkdir -p $R/logs $R/cache $R/results_v2 $R/checkpoints/new

echo "========== 1. 安装依赖（每次新容器都要装）=========="
pip install -q pytorch-lightning sentencepiece av facexlib scikit-image inflect jiwer \
    omegaconf hydra-core fastapi uvicorn python-multipart pandas pyarrow huggingface_hub gdown 2>&1 | tail -2

echo ""
echo "========== 2. GPU 和磁盘 =========="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
df -h $R | tail -1
echo "（数据盘至少要剩 60 GB）"

echo ""
echo "========== 3. 代码仓库 =========="
for d in $R/code/raven $R/code/auto_avsr; do
  [ -d "$d" ] && echo "OK  $d" || echo "缺少 $d"
done
ls $R/code/auto_avsr/preparation/detectors/retinaface/ 2>/dev/null | head

echo ""
echo "========== 4. 权重（把文件名填进 models.csv 的 FILL_ 位置）=========="
ls -lh $R/checkpoints/*.pth 2>/dev/null
echo "--- raven 文件夹 ---"
ls -lh $R/checkpoints/raven/ 2>/dev/null
echo "（BRAVEn 应该是 vsr_prelrs3vox2avs_large_ftlrs3trainvalvox2avs_selftrain_braven.pth；"
echo "  语言模型通常是另一个约几百 MB 的 .pth，名字里可能有 lm 或 language）"

echo ""
echo "========== 5. 数据 =========="
ls $R/data/lrs3_test_hf/data/ 2>/dev/null
echo "--- BBC manifest（找 clip_id,path,text 这种 CSV）---"
find $R -maxdepth 4 -name "*.csv" \( -iname "*bbc*" -o -iname "*manifest*" \) 2>/dev/null | grep -v results | head
echo "--- BBC 视频文件夹 ---"
find $R -maxdepth 4 -type d -iname "*bbc*" 2>/dev/null | head

echo ""
echo "========== 6. 备份 facexlib 权重到持久盘 =========="
FX=$(python -c "import facexlib,os;print(os.path.join(os.path.dirname(facexlib.__file__),'weights'))" 2>/dev/null)
if [ -d "$R/facexlib_weights" ] && [ -n "$FX" ]; then
  mkdir -p "$FX" && cp -n $R/facexlib_weights/* "$FX"/ 2>/dev/null && echo "已从备份恢复 facexlib 权重"
elif [ -d "$FX" ]; then
  mkdir -p $R/facexlib_weights && cp -n "$FX"/* $R/facexlib_weights/ 2>/dev/null && echo "已备份 facexlib 权重"
fi

echo ""
echo "========== 7. 语法检查 =========="
cd "$(dirname "$0")"
for f in common.py prep_lrs3_cache.py cache_bbc_rois.py eval_raven_family.py eval_autoavsr_cache.py run_queue.py; do
  python -m py_compile $f && echo "OK  $f" || echo "错误 $f"
done
echo ""
echo "完成。把第 4、5 部分的输出发给 Claude，确认文件名和路径。"
