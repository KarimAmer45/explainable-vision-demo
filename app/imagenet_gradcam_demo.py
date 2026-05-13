from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
import streamlit as st
import torch
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xai_vision_demo.explain import GradCAM, overlay_heatmap  # noqa: E402


st.set_page_config(
    page_title="GradCAM Image Inspector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --xai-bg: #080b0e;
        --xai-panel: rgba(18, 23, 29, 0.78);
        --xai-panel-strong: rgba(22, 29, 36, 0.92);
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
            linear-gradient(180deg, rgba(242, 189, 107, 0.045), rgba(85, 215, 194, 0.025) 48%, transparent),
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
    div[data-testid="stFileUploader"] section {
        min-height: 150px;
        border: 1px dashed var(--xai-border-strong);
        border-radius: 8px;
        background:
            linear-gradient(135deg, rgba(85, 215, 194, 0.08), rgba(242, 189, 107, 0.035)),
            rgba(7, 11, 15, 0.72);
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: rgba(242, 189, 107, 0.82);
    }
    div[data-testid="stFileUploader"] button {
        border-radius: 6px;
        background: rgba(244, 247, 251, 0.08);
        border-color: rgba(244, 247, 251, 0.18);
        font-weight: 700;
    }
    div[data-testid="stFileUploader"] button:hover {
        border-color: var(--xai-teal);
        color: var(--xai-text);
    }
    div[data-testid="stSlider"] [role="slider"] {
        background: var(--xai-teal);
        border-color: var(--xai-teal);
        box-shadow: 0 0 0 4px rgba(85, 215, 194, 0.14);
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: rgba(7, 11, 15, 0.78);
        border-color: rgba(233, 238, 245, 0.14);
        border-radius: 7px;
    }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--xai-teal), var(--xai-amber));
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
        max-width: 680px;
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
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_imagenet_model() -> tuple[torch.nn.Module, models.ResNet18_Weights]:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    return model, weights


def predict_topk(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_names: list[str],
    top_k: int,
) -> list[tuple[int, str, float]]:
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    scores, indexes = probabilities.topk(top_k)
    return [
        (int(index), class_names[int(index)], float(score))
        for index, score in zip(indexes, scores, strict=True)
    ]


@st.cache_data
def create_sample_image() -> Image.Image:
    image = Image.new("RGB", (960, 640), "#20262a")
    draw = ImageDraw.Draw(image)

    for y in range(640):
        tone = int(28 + y * 0.035)
        draw.line([(0, y), (960, y)], fill=(tone, tone + 6, tone + 9))

    draw.rounded_rectangle((0, 430, 960, 640), radius=0, fill=(42, 32, 26))
    for x in range(0, 960, 34):
        draw.line([(x, 430), (x - 110, 640)], fill=(52, 40, 32), width=2)

    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.ellipse((315, 410, 645, 575), fill=(0, 0, 0, 82))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    image = Image.alpha_composite(image.convert("RGBA"), shadow)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((350, 215, 575, 500), radius=38, fill=(236, 231, 216))
    draw.ellipse((350, 190, 575, 250), fill=(248, 245, 233), outline=(188, 181, 164), width=4)
    draw.ellipse((382, 203, 543, 238), fill=(92, 58, 39))
    draw.rounded_rectangle((385, 240, 540, 478), radius=22, fill=(244, 239, 225))
    draw.arc((530, 285, 685, 430), start=-78, end=85, fill=(235, 229, 214), width=24)
    draw.arc((555, 309, 650, 407), start=-74, end=82, fill=(37, 31, 28), width=16)
    draw.rectangle((395, 246, 538, 326), fill=(248, 244, 233))
    draw.ellipse((396, 198, 540, 238), fill=(81, 50, 34))
    draw.arc((388, 198, 550, 238), start=0, end=180, fill=(148, 112, 83), width=4)

    return image.convert("RGB")


def sample_image_file() -> BytesIO:
    buffer = BytesIO()
    create_sample_image().save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = "sample_coffee_mug.png"
    return buffer


st.markdown('<div class="xai-kicker">Explainable vision demo</div>', unsafe_allow_html=True)
st.title("GradCAM Image Inspector")
st.markdown(
    '<div class="xai-subtitle">A focused model-attention workspace for quick ImageNet '
    "classification checks and GradCAM review.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="xai-pill-row">
        <span class="xai-pill">ResNet18</span>
        <span class="xai-pill">ImageNet</span>
        <span class="xai-pill">GradCAM</span>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_col, filter_col = st.columns([0.62, 0.38], gap="large", vertical_alignment="top")
with upload_col:
    with st.container(border=True):
        st.markdown('<div class="xai-panel-title">Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if st.button("Use sample image"):
            st.session_state["use_sample_image"] = True

with filter_col:
    with st.container(border=True):
        st.markdown('<div class="xai-panel-title">Filters</div>', unsafe_allow_html=True)
        top_k = st.slider("Top predictions", min_value=3, max_value=10, value=5)
        heatmap_alpha = st.slider("Heatmap opacity", min_value=0.15, max_value=0.75, value=0.42)
        target_slot = st.empty()

if uploaded is None:
    if not st.session_state.get("use_sample_image", False):
        st.markdown(
            '<div class="xai-empty">Waiting for an image.</div>',
            unsafe_allow_html=True,
        )
        st.stop()
else:
    st.session_state["use_sample_image"] = False

try:
    with st.spinner("Loading model and generating GradCAM..."):
        model, weights = load_imagenet_model()
        class_names = weights.meta["categories"]
        preprocess = weights.transforms()

        image_file = sample_image_file() if st.session_state.get("use_sample_image", False) else uploaded
        image = Image.open(image_file).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0)
        predictions = predict_topk(model, image_tensor, class_names, top_k)
except Exception as exc:
    st.error(
        "Could not load the pretrained ResNet18 weights. Check your network connection "
        "or run the app again after caching the weights locally."
    )
    st.exception(exc)
    st.stop()

target_options = {
    f"{label} ({probability:.1%})": class_index for class_index, label, probability in predictions
}
with target_slot.container():
    selected_target = st.selectbox("GradCAM target", list(target_options))
target_class = target_options[selected_target]

with st.spinner("Rendering explanation..."):
    gradcam = GradCAM(model, model.layer4[-1])
    try:
        heatmap = gradcam(image_tensor, class_index=target_class)
    finally:
        gradcam.close()

crop_size = weights.transforms().crop_size
overlay_size = crop_size[0] if isinstance(crop_size, list | tuple) else crop_size
resized_image = image.resize((overlay_size, overlay_size))
overlay = overlay_heatmap(resized_image, heatmap, alpha=heatmap_alpha)

left, right = st.columns([1, 1])
with left:
    with st.container(border=True):
        st.subheader("Input")
        st.image(image, use_container_width=True)
with right:
    with st.container(border=True):
        st.subheader("GradCAM")
        st.image(overlay, use_container_width=True)

with st.container(border=True):
    st.subheader("Prediction Confidence")
    for rank, (_class_index, label, probability) in enumerate(predictions, start=1):
        label_col, score_col = st.columns([0.78, 0.22], vertical_alignment="center")
        label_col.markdown(f"**{rank}. {label}**")
        score_col.markdown(f"{probability:.2%}")
        st.progress(min(max(probability, 0.0), 1.0))
