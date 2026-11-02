#!/bin/bash
# 开机后一键启动唇语识别服务（录 Live test 演示时用）
# 运行：bash start_services.sh
# 注意：6007 被 AutoDL 的 TensorBoard 占用，所以 BRAVEn 用 6010

R=/root/autodl-tmp
cd "$(dirname "$0")"
mkdir -p $R/logs

echo "== 1. 安装依赖（新容器需要）"
pip install -q pytorch-lightning sentencepiece av facexlib scikit-image inflect omegaconf hydra-core \
    torchmetrics fastapi uvicorn python-multipart 2>&1 | grep -v WARNING | tail -2

echo "== 2. 恢复 facexlib 权重"
FX=$(python -c "import facexlib,os;print(os.path.join(os.path.dirname(facexlib.__file__),'weights'))" 2>/dev/null)
[ -d "$R/facexlib_weights" ] && mkdir -p "$FX" && cp -n $R/facexlib_weights/* "$FX"/ 2>/dev/null

echo "== 3. 启动服务"
pkill -9 -f lipread_service.py 2>/dev/null; sleep 2
nohup python lipread_service.py --model T1  --port 6006 > $R/logs/svc_T1.log 2>&1 &
nohup python lipread_service.py --model T3  --port 6010 > $R/logs/svc_T3.log 2>&1 &
nohup python lipread_service.py --model U4g --port 6008 > $R/logs/svc_U4g.log 2>&1 &

echo "== 4. 等待加载（最多 3 分钟）"
for i in $(seq 1 36); do
  ok=0
  for p in 6006 6010 6008; do curl -s -m 2 http://127.0.0.1:$p/health > /dev/null && ok=$((ok+1)); done
  echo "  $((i*5)) 秒：$ok/3 个服务就绪"
  [ $ok -eq 3 ] && break
  sleep 5
done

echo ""
for p in 6006 6010 6008; do
  printf "  端口 %s: " $p; curl -s -m 2 http://127.0.0.1:$p/health || echo "未就绪，看 logs/svc_*.log"; echo
done
echo ""
echo "本地隧道命令（新开 Anaconda Prompt 运行，窗口不要关）："
echo "  ssh -CNg -L 6006:127.0.0.1:6006 -L 6007:127.0.0.1:6010 -L 6008:127.0.0.1:6008 root@connect.westc.seetacloud.com -p 28038"
echo "（如果 AutoDL 的 SSH 端口或地址变了，按控制台显示的修改）"
