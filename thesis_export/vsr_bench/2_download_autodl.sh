#!/bin/bash
# 第 2 步：在 AutoDL 上运行
# 作用：从 HuggingFace（hf-mirror）下载全部新权重到 /root/autodl-tmp/checkpoints/new/
#
# 运行前：
#   export HF_TOKEN=hf_xxxxxxxx      # 你的 HF token（私有库需要）
# 运行：
#   bash 2_download_autodl.sh

set -e

# 学术加速代理会挡住 HuggingFace，所以先关掉，再用 hf-mirror
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT=https://hf-mirror.com

CKPT_NEW=/root/autodl-tmp/checkpoints/new
CKPT_AUTO=/root/autodl-tmp/checkpoints
mkdir -p "$CKPT_NEW" /root/autodl-tmp/checkpoints/dllm

if [ -z "$HF_TOKEN" ]; then
  echo "请先运行: export HF_TOKEN=你的token"
  exit 1
fi

python -u << 'EOF'
import os
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files

token = os.environ["HF_TOKEN"]
new_dir = "/root/autodl-tmp/checkpoints/new"

# 1) 你的私有库：new/ 文件夹里的所有权重
files = [f for f in list_repo_files("PEIROU/raven-ckpt", token=token) if f.startswith("new/")]
print("私有库中的新权重:", files)
for f in files:
    target = os.path.join(new_dir, os.path.basename(f))
    if os.path.exists(target):
        print("已存在，跳过:", target); continue
    print("下载:", f)
    p = hf_hub_download("PEIROU/raven-ckpt", f, token=token, local_dir="/root/autodl-tmp/checkpoints")
    print("完成:", p)

# 2) Auto-AVSR S0 / S1（公开库，可能已经在盘上）
for f in ["vsr_trlrs3_base.pth", "vsr_trlrs3vox2_base.pth"]:
    target = os.path.join("/root/autodl-tmp/checkpoints", f)
    if os.path.exists(target):
        print("已存在，跳过:", target); continue
    print("下载:", f)
    hf_hub_download("mpc001/auto_avsr", f, local_dir="/root/autodl-tmp/checkpoints")

# 3) DLLM-VSR（LoRA + 长度预测器，约 240MB）
print("下载 DLLM-VSR ...")
snapshot_download("jh-y/dllm-vsr", local_dir="/root/autodl-tmp/checkpoints/dllm")
print("DLLM-VSR 完成。注意：它还需要 Dream-7B 语言模型，见它的 README。")
EOF

# 4) VSR-ML 仓库的文件都是 zip（S2、S4、S5、两个语言模型），自动解压到同名文件夹
cd "$CKPT_NEW"
for z in *.zip; do
  [ -e "$z" ] || continue
  d="${z%.zip}"; d="${d%.pth}"
  if [ -d "$d" ]; then echo "已解压，跳过: $z"; continue; fi
  echo "解压 $z -> $d"
  unzip -o -q "$z" -d "$d"
done
cd - > /dev/null

echo ""
echo "========== 当前权重 =========="
ls -lh "$CKPT_NEW" "$CKPT_AUTO"/*.pth 2>/dev/null
df -h /root/autodl-tmp | tail -1
