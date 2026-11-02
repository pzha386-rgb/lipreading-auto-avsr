# -*- coding: utf-8 -*-
"""
robot_scenario_gui.py  —  "Robot encounters new users" demo GUI

A storytelling Gradio app showing how a service robot handles 3 new users
in sequence, with the confidence score driving its ACCEPT / ASK AGAIN / REJECT
decision. Independent of lip_compare_gui.py; shares its video + csv data.

Run:
  cd /root/autodl-tmp/vsr_bench/gui
  python robot_scenario_gui.py --port 6007          # pick a free port
  # or
  python robot_scenario_gui.py --port 6007 --share  # try to get a public link
"""
import argparse
import csv
import html
import math
import os

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(HERE, "videos", "multispeakerdemo")
SCENARIO_CSV = os.path.join(HERE, "results", "robot_scenario.csv")

# =============== Decision thresholds (from Sem 1 Tom Scott study) ===============
# confA_logp = per-token log-prob; higher (closer to 0) = more confident
THRESH_ACCEPT = -0.60   # logp > -0.60  -> ACCEPT
THRESH_REJECT = -1.20   # logp < -1.20  -> REJECT; in between -> ASK AGAIN

ROBOT_RESPONSES = {
    "ACCEPT": [
        "Got it — thank you.",
        "Understood, acting on that now.",
        "Thanks, I heard you clearly.",
    ],
    "ASK_AGAIN": [
        "Sorry, I didn't quite catch that — could you repeat it?",
        "I'm not fully sure what you said. Say it again?",
        "Let me check — could you say that one more time?",
    ],
    "REJECT": [
        "I couldn't understand that. Could you speak more clearly, or try again?",
        "I'm having trouble reading your lips. Could you face the camera and repeat?",
        "I can't make that out. Please try again.",
    ],
}


def decide(logp):
    if logp >= THRESH_ACCEPT:
        return "ACCEPT"
    if logp <= THRESH_REJECT:
        return "REJECT"
    return "ASK_AGAIN"


DECISION_COLOR = {
    "ACCEPT":    "#1F8A4C",   # green
    "ASK_AGAIN": "#D08F17",   # amber
    "REJECT":    "#B3263E",   # red
}
DECISION_ICON = {
    "ACCEPT":    "✓",
    "ASK_AGAIN": "⚠",
    "REJECT":    "✗",
}
DECISION_LABEL = {
    "ACCEPT":    "ACCEPT",
    "ASK_AGAIN": "ASK AGAIN",
    "REJECT":    "REJECT",
}


# ============================== Data loading ==============================
def load_scenario():
    """Load the 3 scenario clips in order."""
    rows = []
    with open(SCENARIO_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "step":     int(r["step"]),
                "id":       r["id"],
                "speaker":  r["speaker_display"],
                "ref":      r["ref"],
                "hyp":      r["hyp"],
                "logp":     float(r["confA_logp"]),
                "margin":   float(r["confA_margin"]),
                "seconds":  float(r.get("seconds", 0) or 0),
            })
    rows.sort(key=lambda r: r["step"])
    return rows


# ============================== HTML rendering ==============================
def confidence_bar_html(conf_pct, color):
    """A horizontal bar — SVG, no external deps."""
    pct = max(0, min(100, conf_pct))
    return f"""
    <div style="margin: 6px 0; font-family: -apple-system, sans-serif;">
      <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:4px;">
        <span style="color:#666;">Confidence</span>
        <span style="color:{color}; font-weight:600;">{pct:.0f}%</span>
      </div>
      <div style="width:100%; height:14px; background:#eee; border-radius:7px; overflow:hidden;">
        <div style="width:{pct:.1f}%; height:100%; background:{color}; border-radius:7px; transition:width 0.5s;"></div>
      </div>
    </div>
    """


def decision_banner_html(decision):
    color = DECISION_COLOR[decision]
    icon = DECISION_ICON[decision]
    label = DECISION_LABEL[decision]
    return f"""
    <div style="margin:16px 0; padding:16px 22px; border-radius:10px;
                background:{color}; color:white; font-family: -apple-system, sans-serif;
                display:flex; align-items:center; gap:14px;">
      <span style="font-size:28px; font-weight:700;">{icon}</span>
      <div>
        <div style="font-size:12px; opacity:0.85; letter-spacing:1.5px;">ROBOT DECISION</div>
        <div style="font-size:22px; font-weight:700; letter-spacing:1px;">{label}</div>
      </div>
    </div>
    """


def heard_html(ref, hyp):
    return f"""
    <div style="font-family: -apple-system, sans-serif; margin:6px 0;">
      <div style="color:#666; font-size:13px; margin-bottom:6px;">USER SAID (ground truth)</div>
      <div style="padding:10px 14px; background:#f6f7f9; border-radius:8px; font-size:16px; color:#222;">
        "{html.escape(ref)}"
      </div>
      <div style="color:#666; font-size:13px; margin:12px 0 6px 0;">ROBOT HEARD (model output)</div>
      <div style="padding:10px 14px; background:#eef4ff; border-radius:8px; font-size:16px; color:#1a3d7a;
                  border-left: 4px solid #2F5D8A;">
        {html.escape(hyp) if hyp.strip() else "(nothing)"}
      </div>
    </div>
    """


def robot_response_html(decision, response_text):
    color = DECISION_COLOR[decision]
    return f"""
    <div style="font-family: -apple-system, sans-serif; margin-top:8px;">
      <div style="color:#666; font-size:13px; margin-bottom:6px;">ROBOT RESPONDS</div>
      <div style="padding:12px 16px; background:white; border:2px solid {color};
                  border-radius:10px; font-size:16px; color:#222; font-style:italic;">
        🤖  "{html.escape(response_text)}"
      </div>
    </div>
    """


def step_header_html(step, speaker, total):
    return f"""
    <div style="font-family: -apple-system, sans-serif; margin:4px 0 12px 0;">
      <div style="color:#888; font-size:13px; letter-spacing:2px;">USER {step} OF {total}</div>
      <div style="font-size:26px; font-weight:600; color:#111;">{html.escape(speaker)}</div>
    </div>
    """


# ============================== State machine ==============================
SCENARIO = []   # loaded at startup


def render_step(step_idx):
    """step_idx is 0-based index into SCENARIO."""
    import random
    step_idx = max(0, min(len(SCENARIO) - 1, step_idx))
    clip = SCENARIO[step_idx]
    video = os.path.join(VIDEOS_DIR, f"{clip['id']}.mp4")
    video = video if os.path.exists(video) else None

    # confidence 0-100 from logp (exp maps to 0-1 but logp of -3 is already ~5%)
    conf_pct = math.exp(clip["logp"]) * 100
    decision = decide(clip["logp"])
    color = DECISION_COLOR[decision]
    response = random.choice(ROBOT_RESPONSES[decision])

    header = step_header_html(clip["step"], clip["speaker"], len(SCENARIO))
    heard = heard_html(clip["ref"], clip["hyp"])
    conf = confidence_bar_html(conf_pct, color)
    banner = decision_banner_html(decision)
    resp = robot_response_html(decision, response)

    details = header + heard + conf + banner + resp
    return video, details, step_idx


def next_step(step_idx):
    return render_step((step_idx + 1) % len(SCENARIO))


def prev_step(step_idx):
    return render_step((step_idx - 1) % len(SCENARIO))


def reset():
    return render_step(0)


# ============================== Build UI ==============================
INTRO = """
# 🤖 Robot encounters new users

A service robot meets 3 new users in a row. Each user says one short sentence,
and the robot must decide whether it heard clearly enough to act.

The confidence score comes directly from the lip-reading model (B2, BRAVEn Large + LM).
Thresholds: **ACCEPT** if confidence ≥ 55% · **REJECT** if ≤ 30% · otherwise **ASK AGAIN**.
"""


def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"),
                   css="""
                   .scenario-video video { max-height: 360px; border-radius: 8px; }
                   """) as ui:
        gr.Markdown(INTRO)

        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="User speaking", autoplay=True,
                                 interactive=False, elem_classes="scenario-video")
                with gr.Row():
                    prev_btn = gr.Button("◀ Previous user", size="lg")
                    next_btn = gr.Button("Next user ▶", variant="primary", size="lg")
                reset_btn = gr.Button("↺ Reset to user 1", size="sm")
            with gr.Column(scale=1):
                details = gr.HTML()

        state = gr.State(0)

        # First render on load
        ui.load(render_step, inputs=state, outputs=[video, details, state])

        next_btn.click(next_step, inputs=state, outputs=[video, details, state])
        prev_btn.click(prev_step, inputs=state, outputs=[video, details, state])
        reset_btn.click(reset, outputs=[video, details, state])

    return ui


# ============================== main ==============================
def main():
    global SCENARIO
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6007)
    ap.add_argument("--share", action="store_true", help="try to create a public gradio.live link")
    args = ap.parse_args()

    if not os.path.exists(SCENARIO_CSV):
        raise SystemExit(f"missing {SCENARIO_CSV} — run build_gui_demo_v2.py first")
    SCENARIO = load_scenario()
    print(f"loaded {len(SCENARIO)} scenario clips:")
    for c in SCENARIO:
        decision = decide(c["logp"])
        print(f"  {c['step']}. {c['speaker']:<20} logp={c['logp']:+.2f} -> {decision}")

    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
