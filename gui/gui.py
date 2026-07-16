"""
gui.py — Auto-AVSR Lipreading Test GUI (local Windows) v6
Async task mode + forced English UI

Usage:
    set KMP_DUPLICATE_LIB_OK=TRUE
    python gui.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import requests
import time

try:
    import gradio as gr
except ImportError:
    print("Installing gradio...")
    os.system("pip install gradio")
    import gradio as gr


def check_server(url):
    try:
        url = url.strip().rstrip("/")
        resp = requests.get(f"{url}/health", timeout=10)
        data = resp.json()
        status = f"Connected OK\n"
        status += f"Model loaded: {'yes' if data.get('model_loaded') else 'no (auto-loads on first inference)'}\n"
        status += f"GPU: {data.get('gpu_name', 'N/A')}\n"
        status += f"Weights: {'OK' if data.get('weights_exist') else 'MISSING'}"
        return status
    except requests.exceptions.ConnectionError:
        return "Connection failed - check URL and tunnel"
    except Exception as e:
        return f"Error: {e}"


def preload_model(url):
    try:
        url = url.strip().rstrip("/")
        resp = requests.post(f"{url}/preload", timeout=180)
        data = resp.json()
        return f"Preload: {data.get('message', data.get('status'))}"
    except Exception as e:
        return f"Error: {e}"


def run_inference(video_path, ground_truth, rewrite, extract_roi, url, progress=gr.Progress()):
    if not video_path:
        return "No video", "", ""
    if not url.strip():
        return "Set Server URL first", "", ""

    url = url.strip().rstrip("/")

    try:
        # 1. Submit task (returns task_id quickly)
        progress(0.05, desc="Uploading video...")
        with open(video_path, "rb") as f:
            files = {"file": ("test.mp4", f, "video/mp4")}
            data = {
                "rewrite": str(rewrite).lower(),
                "ground_truth": ground_truth or "",
                "extract_roi": str(extract_roi).lower(),
            }
            resp = requests.post(f"{url}/submit", files=files, data=data, timeout=90)

        task_id = resp.json().get("task_id")
        if not task_id:
            return f"Submit failed: {resp.text[:200]}", "", ""

        # 2. Poll task status
        t0 = time.time()
        max_wait = 1800  # 30 min cap

        while time.time() - t0 < max_wait:
            time.sleep(3)
            elapsed = time.time() - t0
            progress(min(0.05 + elapsed / 300 * 0.9, 0.95),
                     desc=f"Processing... {elapsed:.0f}s (ROI extraction can take 1-5 min)")

            try:
                status_resp = requests.get(f"{url}/status/{task_id}", timeout=30)
                status_data = status_resp.json()
            except Exception:
                continue  # a single failed poll is not fatal

            if status_data.get("status") == "done":
                result = status_data["result"]
                prediction = result.get("prediction", "")
                inference_time = result.get("inference_time", 0)
                frames = result.get("frames", 0)
                wer = result.get("wer")

                total = time.time() - t0
                status = f"Done in {total:.0f}s (inference: {inference_time}s, {frames} frames)"
                wer_str = f"WER: {wer}%" if wer is not None else "No ground truth provided"
                return status, prediction, wer_str

            elif status_data.get("status") == "error":
                msg = status_data["result"].get("message", "unknown")
                return f"Server error: {msg}", "", ""

            elif status_data.get("status") == "not_found":
                return "Task lost - server may have restarted", "", ""

        return "Timeout after 30 minutes", "", ""

    except requests.exceptions.ConnectionError:
        return "Connection failed - tunnel may have expired", "", ""
    except Exception as e:
        return f"Error: {e}", "", ""


# ── Force-English machinery ──
# Layer 1 (head): override navigator.language BEFORE the Gradio frontend
#   initializes, so it picks the English locale from the start.
# Layer 2 (js): text-replacement + MutationObserver as fallback, in case
#   some strings were already rendered from a cached locale.

FORCE_ENGLISH_HEAD = (
    "<script>"
    "try{"
    "Object.defineProperty(navigator,'language',{get:function(){return 'en-US';}});"
    "Object.defineProperty(navigator,'languages',{get:function(){return ['en-US','en'];}});"
    "}catch(e){}"
    "</script>"
)

FORCE_ENGLISH_JS = """
() => {
    const map = {
        "将视频拖放到此处": "Drop video here",
        "将文件拖放到此处": "Drop file here",
        "将图片拖放到此处": "Drop image here",
        "- 或 -": "- or -",
        "或": "or",
        "点击上传": "Click to upload",
        "录制": "Record"
    };
    const walk = (root) => {
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        let node;
        while ((node = walker.nextNode())) {
            const t = node.textContent.trim();
            if (map[t]) node.textContent = map[t];
        }
    };
    walk(document.body);
    new MutationObserver(() => walk(document.body))
        .observe(document.body, {childList: true, subtree: true});
}
"""

with gr.Blocks(title="Auto-AVSR Lipreading Test") as demo:
    gr.Markdown("## Auto-AVSR Lipreading Test GUI")
    gr.Markdown("Upload video → server extracts mouth ROI → VSR inference → prediction")

    with gr.Row():
        with gr.Column(scale=2):
            server_input = gr.Textbox(
                label="Server URL",
                placeholder="https://xxx.trycloudflare.com",
                value=""
            )
            with gr.Row():
                check_btn = gr.Button("Check Connection", size="sm")
                preload_btn = gr.Button("Preload Model", size="sm")
            server_status = gr.Textbox(label="Server Status", lines=4, interactive=False)

            gr.Markdown("---")
            video_input = gr.Video(label="Upload Test Video")
            ground_truth_input = gr.Textbox(
                label="Ground Truth (optional, for WER calculation)",
                placeholder="THE WORDS SPOKEN IN THE VIDEO",
                lines=2
            )
            extract_roi_cb = gr.Checkbox(
                label="Extract mouth ROI on server (for raw videos; ~1-2s per video second)",
                value=True
            )
            rewrite_cb = gr.Checkbox(label="Auto-fix encoding (recommended)", value=True)
            infer_btn = gr.Button("Run Inference", variant="primary", size="lg")

        with gr.Column(scale=3):
            status_output = gr.Textbox(label="Status", lines=2, interactive=False)
            prediction_output = gr.Textbox(label="Prediction", lines=4, interactive=False)
            wer_output = gr.Textbox(label="WER", lines=2, interactive=False)

            gr.Markdown("""
**Tips**
- Raw video (face visible) → keep "Extract mouth ROI" checked
- Pre-cropped 96x96 mouth ROI video → uncheck it
- Async mode: submits task then polls, immune to tunnel timeout
- Keep videos under 30s for reasonable processing time
""")

    check_btn.click(fn=check_server, inputs=server_input, outputs=server_status)
    preload_btn.click(fn=preload_model, inputs=server_input, outputs=server_status)
    infer_btn.click(
        fn=run_inference,
        inputs=[video_input, ground_truth_input, rewrite_cb, extract_roi_cb, server_input],
        outputs=[status_output, prediction_output, wer_output]
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    head=FORCE_ENGLISH_HEAD,
    js=FORCE_ENGLISH_JS,
)
