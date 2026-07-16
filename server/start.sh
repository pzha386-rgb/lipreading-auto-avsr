#!/bin/bash
# start.sh — 一键启动推理服务 + cloudflare 隧道
# 用法: bash /data/default/start.sh

echo "=== 停止旧进程 ==="
kill -9 $(ss -tlnp | grep 8000 | grep -oP 'pid=\K[0-9]+') 2>/dev/null
sleep 1

echo "=== 安装依赖 ==="
pip install fastapi uvicorn python-multipart av rapidfuzz --break-system-packages --quiet 2>/dev/null

echo "=== 启动 server.py ==="
cd /data/default/auto_avsr
python /data/default/server.py &
sleep 5

echo "=== 检查服务 ==="
curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "服务启动中..."

echo ""
echo "=== 启动 cloudflare 隧道 ==="
if [ ! -f /tmp/cloudflared ]; then
    echo "下载 cloudflared..."
    wget -q -O /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x /tmp/cloudflared
fi

/tmp/cloudflared tunnel --url http://localhost:8000 2>&1 | grep -E "trycloudflare|ERR" &
sleep 10

echo ""
echo "=== 完成! ==="
echo "把上面输出的 https://xxx.trycloudflare.com 地址复制到 GUI 里"
echo "按 Ctrl+C 停止"
