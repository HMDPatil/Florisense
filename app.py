import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloriSense",
    
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f0f7f0;
    }

    /* Card container */
    .flower-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    /* Title */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #2d6a4f;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: #74b394;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 2rem;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #2d6a4f, #52b788);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }

    .result-species {
        font-size: 2rem;
        font-weight: 700;
        text-transform: capitalize;
        margin: 0;
    }

    .result-confidence {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }

    /* Upload area */
    .stFileUploader > div {
        border: 2px dashed #52b788 !important;
        border-radius: 12px !important;
        background-color: #f8fdf8 !important;
    }

    /* Button */
    .stButton > button {
        background-color: #2d6a4f;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #1b4332;
    }

    /* Progress bars */
    .stProgress > div > div {
        background-color: #52b788;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
CLASS_EMOJI = {'daisy': '🌼', 'dandelion': '🌻', 'roses': '🌹', 'sunflowers': '🌻', 'tulips': '🌷'}
IMG_SIZE = 224

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('flower_classifier.keras')

model = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🌸 FloriSense</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deep Learning based Flower Species Classifier</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"],
    help="Supports: Daisy, Dandelion, Roses, Sunflowers, Tulips"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button("🔍 Classify"):
            with st.spinner("Analysing..."):
                # Preprocess
                img = image.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                preds = model.predict(img_array, verbose=0)
                pred_idx = np.argmax(preds)
                pred_class = CLASS_NAMES[pred_idx]
                confidence = float(np.max(preds)) * 100
                emoji = CLASS_EMOJI.get(pred_class, "🌸")

            # Result
            st.markdown(f"""
            <div class="result-box">
                <p class="result-species">{emoji} {pred_class}</p>
                <p class="result-confidence">Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("#### All class probabilities")
            for i, (cls, prob) in enumerate(zip(CLASS_NAMES, preds[0])):
                col_name, col_bar = st.columns([1, 3])
                with col_name:
                    st.write(f"{CLASS_EMOJI.get(cls,'🌸')} {cls}")
                with col_bar:
                    st.progress(float(prob))
                    st.caption(f"{prob*100:.1f}%")

# ── Info footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>MobileNetV2 · Transfer Learning · TensorFlow · "
    "TE ECE Internship Project 2025-26</small></center>",
    unsafe_allow_html=True
)
