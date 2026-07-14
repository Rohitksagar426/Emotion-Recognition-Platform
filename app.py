"""
Emotion Console — Real-Time Emotion Recognition
-------------------------------------------------
An instrument-panel style Streamlit app for live webcam emotion
inference using a custom-trained PyTorch EmotionCNN.

Run with:
    streamlit run app.py
"""

import time
from collections import deque

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

st.write("OpenCV Version:", getattr(cv2, "__version__", "No version"))
st.write("cv2 file:", getattr(cv2, "__file__", "No file"))
st.write("Has CascadeClassifier:", hasattr(cv2, "CascadeClassifier"))
st.write("Has data:", hasattr(cv2, "data"))
st.write("dir(cv2) first 30:", dir(cv2)[:30])

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Emotion Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# THEME — instrument-panel / signal-reader aesthetic
# ==============================================================================
BG = "#0B0E14"
PANEL = "#12161F"
PANEL_BORDER = "#232937"
AMBER = "#F5A623"
AMBER_DIM = "#8A611E"
TEXT = "#E8E6E1"
MUTED = "#8B93A1"

EMOTION_COLORS = {
    "Angry": "#E5484D",
    "Disgust": "#6B8E4E",
    "Fear": "#9B6BCE",
    "Happy": "#F5A623",
    "Neutral": "#8B93A1",
    "Sad": "#4FA8D8",
    "Surprise": "#E85D9E",
}

EMOJI = {
    "Angry": "▲",
    "Disgust": "◐",
    "Fear": "◇",
    "Happy": "●",
    "Neutral": "—",
    "Sad": "▽",
    "Surprise": "✦",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'JetBrains Mono', monospace;
}}

.stApp {{
    background: {BG};
    color: {TEXT};
}}

section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {PANEL_BORDER};
}}

h1, h2, h3, h4 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em;
}}

/* Nameplate header */
.nameplate {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 24px;
    background: linear-gradient(180deg, {PANEL} 0%, {BG} 100%);
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
    margin-bottom: 18px;
}}
.nameplate-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: 0.04em;
}}
.nameplate-sub {{
    font-size: 12px;
    color: {MUTED};
    margin-top: 2px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.status-chip {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid {PANEL_BORDER};
    font-size: 12px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
.dot {{
    width: 9px;
    height: 9px;
    border-radius: 50%;
}}
.dot-live {{
    background: #3ECF8E;
    box-shadow: 0 0 8px #3ECF8E;
    animation: pulse 1.4s infinite;
}}
.dot-idle {{
    background: {MUTED};
}}
@keyframes pulse {{
    0% {{ opacity: 1; }}
    50% {{ opacity: 0.35; }}
    100% {{ opacity: 1; }}
}}

/* Panels */
.panel {{
    background: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 16px;
}}
.panel-label {{
    font-size: 11px;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 10px;
}}

/* Video frame bezel */
.video-bezel {{
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
    padding: 8px;
    background: #05070B;
}}

/* Big readout */
.readout-emoji {{
    font-size: 40px;
    line-height: 1;
}}
.readout-label {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px;
    font-weight: 700;
    letter-spacing: 0.02em;
}}
.readout-conf {{
    font-size: 13px;
    color: {MUTED};
    margin-top: 4px;
}}

/* Stat chips */
.stat-chip {{
    background: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}}
.stat-value {{
    font-size: 20px;
    font-weight: 600;
    color: {AMBER};
}}
.stat-label {{
    font-size: 10px;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

/* History strip */
.history-strip {{
    display: flex;
    gap: 3px;
    flex-wrap: wrap;
    align-items: center;
    min-height: 22px;
}}
.tick {{
    width: 10px;
    height: 18px;
    border-radius: 2px;
}}

hr {{
    border-color: {PANEL_BORDER};
}}

div[data-testid="stButton"] > button {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    letter-spacing: 0.03em;
    border-radius: 8px;
    padding: 10px 0px;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ==============================================================================
# MODEL DEFINITION — must match training architecture exactly
# ==============================================================================
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# IMPORTANT: verify against `print(train_dataset.classes)` from training.
# ImageFolder sorts class folders alphabetically — this order must match.
CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


@st.cache_resource(show_spinner="Loading model weights...")
def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


@st.cache_resource(show_spinner="Loading face detector...")
def load_face_cascade():
    BASE_DIR = Path(__file__).parent

    cascade_path = (
        BASE_DIR
        / "haarcascades"
        / "haarcascade_frontalface_default.xml"
    )

    face_cascade = cv2.CascadeClassifier(str(cascade_path))

    if face_cascade.empty():
        raise RuntimeError(
            f"Failed to load Haar Cascade from {cascade_path}"
        )

    return face_cascade


def preprocess_face(face_gray: np.ndarray) -> torch.Tensor:
    face = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    face = face.astype(np.float32) / 255.0
    return torch.from_numpy(face).unsqueeze(0).unsqueeze(0)  # (1,1,48,48)


def predict(model, device, face_gray: np.ndarray):
    tensor = preprocess_face(face_gray).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], probs


def prob_bar_chart(probs_dict: dict, top_class: str):
    df = pd.DataFrame({"Emotion": list(probs_dict.keys()), "Probability": list(probs_dict.values())})
    df["Color"] = df["Emotion"].apply(
        lambda e: EMOTION_COLORS[e] if e == top_class else "#3A4152"
    )
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Emotion:N", sort=CLASSES, axis=alt.Axis(labelColor=MUTED, titleColor=MUTED, labelFontSize=11)),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(labelColor=MUTED, titleColor=MUTED)),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Emotion", alt.Tooltip("Probability", format=".2%")],
        )
        .properties(height=220, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False, domainColor=PANEL_BORDER)
    )
    return chart


# ==============================================================================
# SESSION STATE
# ==============================================================================
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=50)
if "tally" not in st.session_state:
    st.session_state.tally = {c: 0 for c in CLASSES}


def toggle_camera():
    st.session_state.camera_on = not st.session_state.camera_on


# ==============================================================================
# SIDEBAR — SETTINGS
# ==============================================================================
with st.sidebar:
    st.markdown("### ◈ Console Settings")

    with st.expander("Model", expanded=True):
        model_path = st.text_input("Weights file (.pth)", value="best_model.pth")
        camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)

    with st.expander("Face Detection", expanded=True):
        scale_factor = st.slider("Scale factor", 1.05, 1.5, 1.2, step=0.05)
        min_neighbors = st.slider("Min neighbors", 3, 15, 8)
        min_face_size = st.slider("Min face size (px)", 30, 200, 60, step=10)

    with st.expander("Display", expanded=True):
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.4, step=0.05)
        mirror_view = st.checkbox("Mirror (selfie view)", value=True)
        show_history = st.checkbox("Show signal history strip", value=True)

    st.markdown("---")
    st.caption(
        "⚠️ CLASSES must match `train_dataset.classes` exactly — "
        "ImageFolder sorts label folders alphabetically."
    )

# ==============================================================================
# HEADER — NAMEPLATE
# ==============================================================================
status_dot_class = "dot-live" if st.session_state.camera_on else "dot-idle"
status_text = "LIVE" if st.session_state.camera_on else "IDLE"

st.markdown(
    f"""
    <div class="nameplate">
        <div>
            <div class="nameplate-title">◈ EMOTION CONSOLE</div>
            <div class="nameplate-sub">Real-time facial affect reader · PyTorch CNN</div>
        </div>
        <div class="status-chip">
            <span class="dot {status_dot_class}"></span>
            {status_text}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load resources
face_cascade = load_face_cascade()
model, device = None, None
try:
    model, device = load_model(model_path)
except FileNotFoundError:
    st.error(f"Model file not found at '{model_path}'. Update the path in the sidebar.")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# ==============================================================================
# MAIN LAYOUT
# ==============================================================================
col_video, col_stats = st.columns([2.3, 1.2])

with col_video:
    btn_label = "⏹  STOP CAMERA" if st.session_state.camera_on else "▶  START CAMERA"
    btn_type = "secondary" if st.session_state.camera_on else "primary"
    st.button(btn_label, on_click=toggle_camera, use_container_width=True, type=btn_type)

    st.markdown('<div class="video-bezel">', unsafe_allow_html=True)
    FRAME_WINDOW = st.image([])
    st.markdown("</div>", unsafe_allow_html=True)

    stat_a, stat_b, stat_c = st.columns(3)
    fps_stat = stat_a.empty()
    faces_stat = stat_b.empty()
    frames_stat = stat_c.empty()

    if show_history:
        st.markdown('<div class="panel-label" style="margin-top:14px;">SIGNAL HISTORY (last 50 reads)</div>', unsafe_allow_html=True)
        history_strip = st.empty()

with col_stats:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">CURRENT READ</div>', unsafe_allow_html=True)
    emotion_display = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">CLASS PROBABILITIES</div>', unsafe_allow_html=True)
    prob_chart_holder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">SESSION TALLY</div>', unsafe_allow_html=True)
    tally_holder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)


def render_idle_stats():
    emotion_display.markdown(
        f'<div class="readout-emoji" style="color:{MUTED};">—</div>'
        f'<div class="readout-label" style="color:{MUTED};">Standby</div>'
        f'<div class="readout-conf">No signal · start the camera</div>',
        unsafe_allow_html=True,
    )
    prob_chart_holder.altair_chart(
        prob_bar_chart({c: 0.0 for c in CLASSES}, top_class=""), use_container_width=True
    )
    tally_df = pd.DataFrame({"Emotion": CLASSES, "Count": [st.session_state.tally[c] for c in CLASSES]})
    tally_holder.bar_chart(tally_df.set_index("Emotion"), height=140)


def render_history_strip():
    if not st.session_state.history:
        history_strip.markdown(
            f'<div class="history-strip"><span style="color:{MUTED};font-size:12px;">— no reads yet —</span></div>',
            unsafe_allow_html=True,
        )
        return
    ticks = "".join(
        f'<div class="tick" style="background:{EMOTION_COLORS.get(e, MUTED)};" title="{e}"></div>'
        for e in st.session_state.history
    )
    history_strip.markdown(f'<div class="history-strip">{ticks}</div>', unsafe_allow_html=True)


# ==============================================================================
# CAMERA LOOP
# ==============================================================================
if st.session_state.camera_on and model is not None:
    cap = cv2.VideoCapture(int(camera_index))

    if not cap.isOpened():
        st.error(f"Could not open camera index {camera_index}. Try a different index in the sidebar.")
        st.session_state.camera_on = False
    else:
        prev_time = time.time()
        frame_count = 0
        try:
            while st.session_state.camera_on:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to read frame from camera.")
                    break

                if mirror_view:
                    frame = cv2.flip(frame, 1)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(min_face_size, min_face_size),
                )

                label, probs = None, None

                for (x, y, w, h) in faces:
                    face_gray = gray[y:y + h, x:x + w]
                    label, probs = predict(model, device, face_gray)
                    confidence = float(np.max(probs))
                    color_hex = EMOTION_COLORS.get(label, "#3ECF8E")
                    b, g, r = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
                    box_color = (b, g, r)
                    display_label = f"{label} {confidence*100:.0f}%"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(frame, display_label, (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                    if confidence >= confidence_threshold:
                        st.session_state.history.append(label)
                        st.session_state.tally[label] += 1

                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time
                frame_count += 1

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)

                fps_stat.markdown(f'<div class="stat-chip"><div class="stat-value">{fps:.1f}</div><div class="stat-label">FPS</div></div>', unsafe_allow_html=True)
                faces_stat.markdown(f'<div class="stat-chip"><div class="stat-value">{len(faces)}</div><div class="stat-label">Faces</div></div>', unsafe_allow_html=True)
                frames_stat.markdown(f'<div class="stat-chip"><div class="stat-value">{frame_count}</div><div class="stat-label">Frames</div></div>', unsafe_allow_html=True)

                if label is not None and probs is not None:
                    top_conf = float(np.max(probs))
                    emo_color = EMOTION_COLORS.get(label, TEXT)
                    emotion_display.markdown(
                        f'<div class="readout-emoji" style="color:{emo_color};">{EMOJI.get(label,"")}</div>'
                        f'<div class="readout-label" style="color:{emo_color};">{label}</div>'
                        f'<div class="readout-conf">{top_conf*100:.1f}% confidence</div>',
                        unsafe_allow_html=True,
                    )
                    probs_dict = {c: float(p) for c, p in zip(CLASSES, probs)}
                    prob_chart_holder.altair_chart(prob_bar_chart(probs_dict, label), use_container_width=True)
                else:
                    render_idle_stats()
                    emotion_display.markdown(
                        f'<div class="readout-emoji" style="color:{MUTED};">—</div>'
                        f'<div class="readout-label" style="color:{MUTED};">No face</div>'
                        f'<div class="readout-conf">Searching for a face...</div>',
                        unsafe_allow_html=True,
                    )

                tally_df = pd.DataFrame({"Emotion": CLASSES, "Count": [st.session_state.tally[c] for c in CLASSES]})
                tally_holder.bar_chart(tally_df.set_index("Emotion"), height=140)

                if show_history:
                    render_history_strip()

        finally:
            cap.release()

elif st.session_state.camera_on and model is None:
    st.warning("Fix the model loading error above, then press Start Camera again.")
    st.session_state.camera_on = False
else:
    render_idle_stats()
    if show_history:
        render_history_strip()