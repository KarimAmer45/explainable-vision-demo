from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Transformer Attention Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --xai-bg: #080b0e;
        --xai-panel: rgba(18, 23, 29, 0.78);
        --xai-border: rgba(233, 238, 245, 0.12);
        --xai-border-strong: rgba(84, 206, 188, 0.38);
        --xai-text: #f4f7fb;
        --xai-muted: rgba(229, 235, 243, 0.68);
        --xai-teal: #55d7c2;
        --xai-amber: #f2bd6b;
    }
    html, body, [class*="css"] {
        font-family:
            "Inter", "Aptos", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container {
        max-width: 1160px;
        padding-top: 3.25rem;
        padding-bottom: 3rem;
    }
    .stApp {
        background:
            linear-gradient(145deg, rgba(28, 39, 46, 0.78), rgba(8, 11, 14, 0.98) 38%),
            linear-gradient(
                180deg,
                rgba(242, 189, 107, 0.045),
                rgba(85, 215, 194, 0.025) 48%,
                transparent
            ),
            var(--xai-bg);
    }
    h1 {
        color: var(--xai-text) !important;
        font-size: 2.6rem !important;
        font-weight: 760 !important;
        line-height: 1.04 !important;
        letter-spacing: 0 !important;
        margin-bottom: 0.45rem !important;
    }
    h2, h3 {
        color: var(--xai-text) !important;
        font-weight: 720 !important;
        letter-spacing: 0 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.018)),
            var(--xai-panel);
        border: 1px solid var(--xai-border);
        border-radius: 8px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
    }
    div[data-testid="stTextInput"] input {
        background: rgba(7, 11, 15, 0.78);
        border-color: rgba(233, 238, 245, 0.14);
        border-radius: 7px;
        color: var(--xai-text);
    }
    div[data-testid="stImage"] img {
        border-radius: 7px;
    }
    .xai-kicker {
        color: var(--xai-teal);
        font-size: 0.78rem;
        font-weight: 760;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .xai-subtitle {
        color: var(--xai-muted);
        font-size: 1.02rem;
        line-height: 1.56;
        margin-bottom: 1.2rem;
        max-width: 760px;
    }
    .xai-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.85rem 0 1.25rem;
    }
    .xai-pill {
        border: 1px solid rgba(233, 238, 245, 0.16);
        border-radius: 999px;
        color: rgba(244, 247, 251, 0.9);
        background: rgba(244, 247, 251, 0.055);
        padding: 0.34rem 0.78rem;
        font-size: 0.82rem;
        font-weight: 650;
    }
    .xai-panel-title {
        color: var(--xai-text);
        font-size: 0.98rem;
        font-weight: 760;
        margin-bottom: 0.45rem;
    }
    .xai-empty {
        border: 1px solid var(--xai-border);
        border-radius: 8px;
        padding: 1.2rem;
        color: var(--xai-muted);
        background: rgba(244, 247, 251, 0.035);
    }
    .xai-metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.8rem;
    }
    .xai-metric {
        border: 1px solid rgba(233, 238, 245, 0.12);
        border-radius: 8px;
        background: rgba(7, 11, 15, 0.58);
        padding: 0.85rem 0.95rem;
    }
    .xai-metric-label {
        color: var(--xai-muted);
        font-size: 0.78rem;
        font-weight: 680;
        margin-bottom: 0.3rem;
    }
    .xai-metric-value {
        color: var(--xai-text);
        font-size: 1.35rem;
        font-weight: 780;
    }
    .xai-command {
        border: 1px solid rgba(84, 206, 188, 0.28);
        border-radius: 8px;
        color: rgba(244, 247, 251, 0.92);
        background: rgba(7, 11, 15, 0.78);
        padding: 0.9rem 1rem;
        font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
        font-size: 0.86rem;
        line-height: 1.55;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_card(label: str, value: float | int | str) -> str:
    if isinstance(value, float):
        rendered = f"{value:.3f}"
    else:
        rendered = str(value)
    return (
        '<div class="xai-metric">'
        f'<div class="xai-metric-label">{label}</div>'
        f'<div class="xai-metric-value">{rendered}</div>'
        "</div>"
    )


st.markdown('<div class="xai-kicker">Transformer experiment</div>', unsafe_allow_html=True)
st.title("Transformer Attention Lab")
st.markdown(
    '<div class="xai-subtitle">A compact model-audit surface for the CIFAR-10 ViT/Swin '
    "experiment, with metric cards, run provenance, and a transformer-native attention "
    "rollout artifact.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="xai-pill-row">
        <span class="xai-pill">CIFAR-10</span>
        <span class="xai-pill">ViT / Swin</span>
        <span class="xai-pill">Attention rollout</span>
    </div>
    """,
    unsafe_allow_html=True,
)

control_col, summary_col = st.columns([0.38, 0.62], gap="large", vertical_alignment="top")
with control_col:
    with st.container(border=True):
        st.markdown('<div class="xai-panel-title">Run Directory</div>', unsafe_allow_html=True)
        run_dir_text = st.text_input("Run directory", "runs/cifar10_vit_smoke")
        run_dir = (ROOT / run_dir_text).resolve()
        st.caption(str(run_dir))

with summary_col:
    with st.container(border=True):
        st.markdown('<div class="xai-panel-title">Expected Command</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="xai-command">python -m xai_vision_demo.transformer_experiment ^
  --arch vit_b_16 ^
  --output-dir runs/cifar10_vit_smoke ^
  --epochs 1 ^
  --max-train-samples 100 ^
  --max-val-samples 50 ^
  --max-test-samples 50</div>
            """,
            unsafe_allow_html=True,
        )

required_files = [
    "test_metrics.json",
    "metrics_history.json",
    "classes.json",
    "attention_rollout_example.png",
    "training_curves.png",
]
missing = [name for name in required_files if not (run_dir / name).exists()]
if missing:
    st.markdown(
        f'<div class="xai-empty">Missing run artifacts: {", ".join(missing)}.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

metrics = read_json(run_dir / "test_metrics.json")
history = read_json(run_dir / "metrics_history.json")
classes = read_json(run_dir / "classes.json")
config = read_json(run_dir / "run_config.json") if (run_dir / "run_config.json").exists() else {}
checkpoint = run_dir / "best_model.pt"

with st.container(border=True):
    st.markdown('<div class="xai-panel-title">Test Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="xai-metric-grid">'
        + metric_card("Accuracy", metrics["accuracy"])
        + metric_card("Macro ROC-AUC", metrics["macro_auc_ovr"])
        + metric_card("Macro AP", metrics["macro_average_precision"])
        + metric_card("Loss", metrics["loss"])
        + "</div>",
        unsafe_allow_html=True,
    )

left, right = st.columns([1, 1], gap="large")
with left:
    with st.container(border=True):
        st.subheader("Attention Rollout")
        st.image(Image.open(run_dir / "attention_rollout_example.png"), use_container_width=True)
with right:
    with st.container(border=True):
        st.subheader("Training Curve")
        st.image(Image.open(run_dir / "training_curves.png"), use_container_width=True)

detail_col, class_col = st.columns([0.48, 0.52], gap="large")
with detail_col:
    with st.container(border=True):
        st.subheader("Run Provenance")
        st.markdown(f"**Architecture:** `{config.get('arch', 'vit_b_16')}`")
        st.markdown(f"**Pretrained:** `{config.get('pretrained', True)}`")
        st.markdown(f"**Frozen backbone:** `{config.get('freeze_backbone', True)}`")
        st.markdown(f"**Checkpoint:** `{checkpoint.name}` ({checkpoint.stat().st_size / 1_000_000:.1f} MB)")
with class_col:
    with st.container(border=True):
        st.subheader("Classes")
        class_table = pd.DataFrame({"index": range(len(classes)), "class": classes})
        st.dataframe(class_table, use_container_width=True, hide_index=True)

if history:
    with st.container(border=True):
        st.subheader("Epoch History")
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
