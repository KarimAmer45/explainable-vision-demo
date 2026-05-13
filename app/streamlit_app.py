from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_demo.data import build_transforms  # noqa: E402
from vision_demo.explain import GradCAM, overlay_heatmap  # noqa: E402
from vision_demo.model import create_model, gradcam_target_layer, is_vit_arch  # noqa: E402
from vision_demo.transformer_explain import attention_rollout_heatmap  # noqa: E402


st.set_page_config(
    page_title="Explainable Vision Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --vision-bg: #080b0e;
        --vision-panel: rgba(18, 23, 29, 0.78);
        --vision-border: rgba(233, 238, 245, 0.12);
        --vision-border-strong: rgba(84, 206, 188, 0.38);
        --vision-text: #f4f7fb;
        --vision-muted: rgba(229, 235, 243, 0.68);
        --vision-teal: #55d7c2;
        --vision-amber: #f2bd6b;
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
            var(--vision-bg);
    }
    h1 {
        color: var(--vision-text) !important;
        font-size: 2.6rem !important;
        font-weight: 760 !important;
        line-height: 1.04 !important;
        letter-spacing: 0 !important;
        margin-bottom: 0.45rem !important;
    }
    h2, h3 {
        color: var(--vision-text) !important;
        font-weight: 720 !important;
        letter-spacing: 0 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.018)),
            var(--vision-panel);
        border: 1px solid var(--vision-border);
        border-radius: 8px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
    }
    div[data-testid="stFileUploader"] section {
        min-height: 150px;
        border: 1px dashed var(--vision-border-strong);
        border-radius: 8px;
        background:
            linear-gradient(135deg, rgba(85, 215, 194, 0.08), rgba(242, 189, 107, 0.035)),
            rgba(7, 11, 15, 0.72);
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: rgba(242, 189, 107, 0.82);
    }
    div[data-testid="stFileUploader"] button,
    div[data-testid="stButton"] button {
        border-radius: 6px;
        background: rgba(244, 247, 251, 0.08);
        border-color: rgba(244, 247, 251, 0.18);
        font-weight: 700;
    }
    div[data-testid="stFileUploader"] button:hover,
    div[data-testid="stButton"] button:hover {
        border-color: var(--vision-teal);
        color: var(--vision-text);
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: rgba(7, 11, 15, 0.78);
        border-color: rgba(233, 238, 245, 0.14);
        border-radius: 7px;
        color: var(--vision-text);
    }
    div[data-testid="stSlider"] [role="slider"] {
        background: var(--vision-teal);
        border-color: var(--vision-teal);
        box-shadow: 0 0 0 4px rgba(85, 215, 194, 0.14);
    }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--vision-teal), var(--vision-amber));
    }
    div[data-testid="stImage"] img {
        border-radius: 7px;
    }
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }
    .vision-kicker {
        color: var(--vision-teal);
        font-size: 0.78rem;
        font-weight: 760;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .vision-subtitle {
        color: var(--vision-muted);
        font-size: 1.02rem;
        line-height: 1.56;
        margin-bottom: 1.2rem;
        max-width: 720px;
    }
    .vision-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.85rem 0 1.25rem;
    }
    .vision-pill {
        border: 1px solid rgba(233, 238, 245, 0.16);
        border-radius: 999px;
        color: rgba(244, 247, 251, 0.9);
        background: rgba(244, 247, 251, 0.055);
        padding: 0.34rem 0.78rem;
        font-size: 0.82rem;
        font-weight: 650;
    }
    .vision-panel-title {
        color: var(--vision-text);
        font-size: 0.98rem;
        font-weight: 760;
        margin-bottom: 0.45rem;
    }
    .vision-empty {
        border: 1px solid var(--vision-border);
        border-radius: 8px;
        padding: 1.2rem;
        color: var(--vision-muted);
        background: rgba(244, 247, 251, 0.035);
    }
    .vision-score-label {
        color: var(--vision-text);
        font-weight: 720;
    }
    .vision-score-value {
        color: var(--vision-muted);
        text-align: right;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = create_model(
        checkpoint["arch"],
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


st.markdown('<div class="vision-kicker">Explainable vision demo</div>', unsafe_allow_html=True)
st.title("Explainable Vision Demo")
st.markdown(
    '<div class="vision-subtitle">A clean inspection workspace for local image classifiers, '
    "top-class confidence, and GradCAM or attention-rollout review.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="vision-pill-row">
        <span class="vision-pill">Local checkpoint</span>
        <span class="vision-pill">Classification</span>
        <span class="vision-pill">GradCAM / Rollout</span>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_col, filter_col = st.columns([0.62, 0.38], gap="large", vertical_alignment="top")
with upload_col:
    with st.container(border=True):
        st.markdown('<div class="vision-panel-title">Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

with filter_col:
    with st.container(border=True):
        st.markdown('<div class="vision-panel-title">Model</div>', unsafe_allow_html=True)
        checkpoint_path = st.text_input("Checkpoint", "runs/surface_resnet18/best_model.pt")
        top_predictions = st.slider("Top predictions", min_value=1, max_value=10, value=5)
        heatmap_alpha = st.slider("Heatmap opacity", min_value=0.15, max_value=0.75, value=0.42)
        target_slot = st.empty()


if not Path(checkpoint_path).exists():
    st.markdown(
        '<div class="vision-empty">Enter a valid checkpoint path to start inference.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if uploaded is None:
    st.markdown(
        '<div class="vision-empty">Waiting for an image.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

with st.spinner("Loading model and preparing explanation..."):
    model, checkpoint = load_model(checkpoint_path)
    arch = checkpoint["arch"]
    uses_attention_rollout = is_vit_arch(arch)
    explanation_title = "Attention Rollout" if uses_attention_rollout else "GradCAM"
    class_names = checkpoint["class_names"]
    image = Image.open(uploaded).convert("RGB")
    transform = build_transforms(checkpoint["image_size"], train=False)
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

score_table = (
    pd.DataFrame({"class": class_names, "probability": probs})
    .sort_values("probability", ascending=False)
    .reset_index(drop=True)
)
top_count = min(top_predictions, len(score_table))
target_options = {
    f"{row['class']} ({row['probability']:.1%})": int(
        class_names.index(row["class"])
    )
    for _, row in score_table.head(top_count).iterrows()
}
with target_slot.container():
    selected_target = st.selectbox("Explanation target", list(target_options))
target_class = target_options[selected_target]

with st.spinner("Rendering explanation..."):
    if uses_attention_rollout:
        heatmap = attention_rollout_heatmap(model, image_tensor)
    else:
        try:
            target_layer = gradcam_target_layer(model, arch)
        except ValueError:
            st.error(
                "This checkpoint can run classification, but the UI currently supports "
                "GradCAM for CNNs and attention rollout for ViT-B/16 checkpoints."
            )
            st.stop()
        gradcam = GradCAM(model, target_layer)
        try:
            heatmap = gradcam(image_tensor, class_index=target_class)
        finally:
            gradcam.close()

overlay = overlay_heatmap(
    image.resize((checkpoint["image_size"], checkpoint["image_size"])),
    heatmap,
    alpha=heatmap_alpha,
)

left, right = st.columns([1, 1])
with left:
    with st.container(border=True):
        st.subheader("Input")
        st.image(image, use_container_width=True)
with right:
    with st.container(border=True):
        st.subheader(explanation_title)
        st.image(overlay, use_container_width=True)

with st.container(border=True):
    st.subheader("Prediction Confidence")
    for rank, row in score_table.head(top_count).iterrows():
        label_col, score_col = st.columns([0.78, 0.22], vertical_alignment="center")
        label_col.markdown(
            f'<div class="vision-score-label">{rank + 1}. {row["class"]}</div>',
            unsafe_allow_html=True,
        )
        score_col.markdown(
            f'<div class="vision-score-value">{row["probability"]:.2%}</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(max(float(row["probability"]), 0.0), 1.0))
