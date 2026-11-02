# -*- coding: utf-8 -*-
"""
第 1 步：在新西兰本地电脑运行（Windows / Anaconda Prompt）
作用：从 Google Drive 下载新权重 -> 上传到 HuggingFace 私有库 PEIROU/raven-ckpt 的 new/ 文件夹

准备（只需一次）：
    pip install gdown huggingface_hub requests
    huggingface-cli login        # 粘贴你的 HF token（要有 write 权限）

运行：
    python 1_download_local.py                 # 下载 + 上传全部
    python 1_download_local.py --only R2 U4    # 只处理指定编号
    python 1_download_local.py --no-upload     # 只下载，不上传
"""
import argparse
import os

import gdown
import requests
from huggingface_hub import HfApi

HF_REPO = "PEIROU/raven-ckpt"
HF_SUBDIR = "new"

# 编号: (保存的文件名, Google Drive 链接)
FILES = {
    # ---------- RAVEn（高资源）----------
    "R2": ("R2_raven_high_large_st.pth",
           "https://drive.google.com/file/d/1tNZn_BvAVdoIIv6G14_9PvpQtsp6XSOt/view"),
    # ---------- BRAVEn ----------
    "B1": ("B1_braven_low_large_avs.pth",
           "https://drive.google.com/file/d/1GiSlZwTcCr5ZrZdntOqubZrdcsH7shel/view"),
    "B3": ("B3_braven_high_large_avs.pth",
           "https://drive.google.com/file/d/1gjHOR-zvavxAMk1vTQmkodDT11jXyeJx/view"),
    "B4": ("B4_braven_high_large_avs_st.pth",
           "https://drive.google.com/file/d/1bU-bFxEiXNXNoOLKaAz6_9j7XuE7f_6r/view"),
    # ---------- USR 2.0 ----------
    "U1": ("U1_usr2_low_large.pth",
           "https://drive.google.com/file/d/1FIculf1Mfo73Y2f_dSu-HDw1Jb36pW-r/view"),
    "U2": ("U2_usr2_high_baseplus.pth",
           "https://drive.google.com/file/d/18vmJjdem5XPOA8bmizybIW5sLJuHMdRR/view"),
    "U3": ("U3_usr2_high_large.pth",
           "https://drive.google.com/file/d/1XDNLfZMv_nn8ALiFmDh3asrJKGC2uHs1/view"),
    "U4": ("U4_usr2_high_huge.pth",
           "https://drive.google.com/file/d/1LzFOTYu45zCLOHGVLQt7pMGjw6jmmo9Y/view"),
    # ---------- Auto-AVSR（VSR-ML 仓库，bit.ly 短链接）----------
    # 如果下载失败：用百度网盘 https://bit.ly/3ZjbrV5  提取码 dqsy
    "S2": ("S2_autoavsr_vsrml.pth.zip", "http://bit.ly/40EAtyX"),
    # Auto-AVSR 的语言模型（给 S2+LM 用）  百度备用 https://bit.ly/3yRI5SY 提取码 t9ep
    "LM_AUTO": ("LM_autoavsr.zip", "http://bit.ly/3FE4XsV"),
    # ---------- 更多监督学习模型（VSR-ML 仓库，Ma et al. 2022）----------
    # S4: LRS3 视觉模型  百度备用 https://bit.ly/3rIzLCn 提取码 1b1s
    "S4": ("S4_vsrml_lrs3.zip", "https://bit.ly/3Bp4gjV"),
    # S5: LRS2 视觉模型（跨数据集测试）  百度备用 https://bit.ly/3BAHBkH 提取码 48l1
    "S5": ("S5_vsrml_lrs2.zip", "https://bit.ly/3I25zrH"),
    # VSR-ML 语言模型（给 S4+LM、S5+LM 用）  百度备用 https://bit.ly/3KgAL7T 提取码 59u2
    "LM_VSRML": ("LM_vsrml.zip", "https://bit.ly/3qzWKit"),
    # S6: Auto-AVSR 只用 23 小时标签（对照自监督的 30 小时低资源设置）
    "S6": ("S6_autoavsr_23h.pth",
           "https://drive.google.com/file/d/1FDDTOBteJV8yBiJ8yePtZ-C-xR4s80rV/view"),
}


def drive_id(url):
    """从 Google Drive 链接里取出文件 ID"""
    import re
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", url) or re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", url)
    return m.group(1) if m else None


def resolve_url(url):
    """bit.ly 短链接 -> 真实的 Google Drive 链接"""
    if "bit.ly" not in url:
        return url
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        print(f"  短链接解析为: {r.url}")
        return r.url
    except Exception as e:
        print(f"  短链接解析失败: {e}")
        return url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"D:\vsr_ckpts", help="本地保存文件夹")
    ap.add_argument("--only", nargs="*", help="只处理这些编号，例如 R2 U4")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ids = args.only if args.only else list(FILES.keys())

    api = HfApi()
    existing = set()
    if not args.no_upload:
        try:
            existing = set(api.list_repo_files(HF_REPO))
        except Exception as e:
            print(f"读取 HF 仓库失败（检查是否 login）: {e}")
            return

    report = []
    for mid in ids:
        if mid not in FILES:
            print(f"[跳过] 未知编号 {mid}")
            continue
        fname, url = FILES[mid]
        local = os.path.join(args.out, fname)
        repo_path = f"{HF_SUBDIR}/{fname}"
        print(f"\n===== {mid}: {fname} =====")

        # 1) 下载（已存在就跳过）
        if os.path.exists(local) and os.path.getsize(local) > 1_000_000:
            print("  本地已存在，跳过下载")
        else:
            try:
                fid = drive_id(resolve_url(url))
                if not fid:
                    raise RuntimeError("链接里找不到 Google Drive 文件 ID")
                # 用 uc?id= 形式，所有 gdown 版本都支持（gdown 6 去掉了 fuzzy 参数）
                gdown.download(f"https://drive.google.com/uc?id={fid}", local, quiet=False)
            except Exception as e:
                print(f"  下载失败: {e}")
                report.append((mid, "下载失败"))
                continue
        if not os.path.exists(local) or os.path.getsize(local) < 1_000_000:
            print("  文件太小，可能是网页而不是权重，请手动下载")
            report.append((mid, "文件异常"))
            continue
        size_gb = os.path.getsize(local) / 1e9
        print(f"  大小: {size_gb:.2f} GB")

        # 2) 上传（已存在就跳过）
        if args.no_upload:
            report.append((mid, f"已下载 {size_gb:.2f}GB"))
            continue
        if repo_path in existing:
            print("  HF 上已存在，跳过上传")
            report.append((mid, "已在 HF"))
            continue
        try:
            api.upload_file(path_or_fileobj=local, path_in_repo=repo_path,
                            repo_id=HF_REPO, commit_message=f"add {mid}")
            print("  上传完成")
            report.append((mid, f"上传完成 {size_gb:.2f}GB"))
        except Exception as e:
            print(f"  上传失败: {e}")
            report.append((mid, "上传失败"))

    print("\n========== 汇总 ==========")
    for mid, s in report:
        print(f"{mid:4s} {s}")
    print("\n注意：S2、S4、S5 和两个 LM 都是 zip 压缩包，到 AutoDL 后会自动解压。")


if __name__ == "__main__":
    main()
