import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import time

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="centered"
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .title { font-size: 28px; font-weight: 700; color: #1B2A4A; margin-bottom: 4px; }
    .subtitle { font-size: 14px; color: #64748B; margin-bottom: 24px; }
    .result-box { padding: 20px; border-radius: 8px; margin: 12px 0; }
    .normal-box { background-color: #DCFCE7; border-left: 4px solid #16A34A; }
    .pneumonia-box { background-color: #FEE2E2; border-left: 4px solid #DC2626; }
    .uncertain-box { background-color: #FEF9C3; border-left: 4px solid #CA8A04; }
    .metric-label { font-size: 12px; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1B2A4A; }
    .disclaimer { font-size: 11px; color: #94A3B8; margin-top: 24px; text-align: center; }
    .step { background: #F8FAFC; padding: 10px 14px; border-radius: 6px; margin: 6px 0; font-size: 13px; color: #475569; }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        'best_model.h5',
        compile=False
    )
# ── MC Dropout inference ─────────────────────────────────────────────────────
def mc_dropout_predict(model, img_array, num_runs=30):
    img_tensor = tf.constant(img_array, dtype=tf.float32)
    preds = []
    for _ in range(num_runs):
        pred = model(img_tensor, training=True).numpy()[0][0]
        preds.append(pred)
    mean_pred = np.mean(preds)
    uncertainty = np.std(preds)
    return mean_pred, uncertainty

# ── Preprocess image ─────────────────────────────────────────────────────────
def preprocess(image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="title">🫁 Pneumonia Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Domain-Generalizable CNN-ANN Hybrid with MC Dropout Uncertainty Quantification</div>', unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Upload Chest X-Ray")
    uploaded = st.file_uploader(
        "Select a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a frontal chest X-ray image (PA or AP view)"
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded X-Ray", width="stretch")

with col2:
    st.markdown("#### How it works")
    st.markdown('<div class="step">1. Upload a chest X-ray image</div>', unsafe_allow_html=True)
    st.markdown('<div class="step">2. CNN (MobileNetV2) extracts visual features</div>', unsafe_allow_html=True)
    st.markdown('<div class="step">3. ANN classifier predicts Normal or Pneumonia</div>', unsafe_allow_html=True)
    st.markdown('<div class="step">4. MC Dropout runs 30 passes to estimate uncertainty</div>', unsafe_allow_html=True)
    st.markdown('<div class="step">5. Result + confidence + trust signal shown below</div>', unsafe_allow_html=True)

    st.markdown("#### Model Info")
    st.markdown("""
    - **Architecture:** MobileNetV2 + ANN hybrid
    - **Trained on:** Kermany dataset (5,216 images)
    - **Cross-tested on:** Patel dataset (5,856 images)
    - **Kermany accuracy:** 89.90%
    - **Cross-dataset accuracy:** 90.33%
    """)

st.markdown("---")

if uploaded:
    if st.button("🔍 Analyse X-Ray", use_container_width=True, type="primary"):
        with st.spinner("Loading model..."):
            try:
                model = load_model()
            except Exception as e:
                st.error(f"Model not found. Make sure best_model.h5 is in the same folder. Error: {e}")
                st.stop()

        progress = st.progress(0, text="Preprocessing image...")
        time.sleep(0.3)

        img_array = preprocess(image)
        progress.progress(20, text="Running MC Dropout inference (30 passes)...")

        mean_pred, uncertainty = mc_dropout_predict(model, img_array, num_runs=30)
        progress.progress(100, text="Done!")
        time.sleep(0.3)
        progress.empty()

        # Results
        st.markdown("### Results")

        confidence = mean_pred if mean_pred >= 0.5 else 1 - mean_pred
        prediction = "Pneumonia" if mean_pred >= 0.5 else "Normal"
        is_uncertain = uncertainty > 0.15

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown('<div class="metric-label">Prediction</div>', unsafe_allow_html=True)
            color = "#DC2626" if prediction == "Pneumonia" else "#16A34A"
            st.markdown(f'<div class="metric-value" style="color:{color}">{prediction}</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="metric-label">Confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{confidence:.1%}</div>', unsafe_allow_html=True)

        with col_c:
            st.markdown('<div class="metric-label">Uncertainty</div>', unsafe_allow_html=True)
            unc_color = "#CA8A04" if is_uncertain else "#16A34A"
            st.markdown(f'<div class="metric-value" style="color:{unc_color}">{uncertainty:.4f}</div>', unsafe_allow_html=True)

        st.markdown("")

        # Result box
        if is_uncertain:
            st.markdown(f"""
            <div class="result-box uncertain-box">
                <strong>⚠️ Uncertain Prediction — Refer to Specialist</strong><br>
                The model predicted <strong>{prediction}</strong> but with high uncertainty ({uncertainty:.4f}).
                This X-ray may contain features outside the model's training distribution.
                A radiologist should review this case manually.
            </div>
            """, unsafe_allow_html=True)
        elif prediction == "Pneumonia":
            st.markdown(f"""
            <div class="result-box pneumonia-box">
                <strong>🔴 Pneumonia Detected</strong><br>
                The model predicts <strong>Pneumonia</strong> with {confidence:.1%} confidence
                and low uncertainty ({uncertainty:.4f}). The model is confident in this prediction.
                Clinical correlation and radiologist confirmation is recommended before treatment.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box normal-box">
                <strong>🟢 Normal — No Pneumonia Detected</strong><br>
                The model predicts <strong>Normal</strong> with {confidence:.1%} confidence
                and low uncertainty ({uncertainty:.4f}). The model is confident in this prediction.
                Clinical correlation is always recommended.
            </div>
            """, unsafe_allow_html=True)

        # Probability bar
        st.markdown("#### Prediction Probability")
        st.progress(float(mean_pred), text=f"Pneumonia probability: {mean_pred:.1%}")

        st.markdown("#### MC Dropout Uncertainty Interpretation")
        unc_col1, unc_col2 = st.columns(2)
        with unc_col1:
            st.metric("Uncertainty Score", f"{uncertainty:.4f}")
            if uncertainty < 0.08:
                st.success("Very low uncertainty — highly confident")
            elif uncertainty < 0.15:
                st.info("Moderate uncertainty — reasonably confident")
            else:
                st.warning("High uncertainty — refer to specialist")
        with unc_col2:
            st.metric("MC Dropout Runs", "30")
            st.metric("Raw Prediction Score", f"{mean_pred:.4f}")

else:
    st.info("Upload a chest X-ray image above to get started.")

st.markdown("""
<div class="disclaimer">
⚕️ This tool is for research and educational purposes only. It is not a substitute for professional medical diagnosis.
All predictions must be verified by a qualified radiologist before any clinical decision is made.<br><br>
Project: Domain-Generalizable CNN-ANN Hybrid Framework for Cross-Dataset Pneumonia Detection with Uncertainty Quantification
</div>
""", unsafe_allow_html=True)
