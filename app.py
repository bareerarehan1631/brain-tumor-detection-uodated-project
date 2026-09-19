import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "best.pt"

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

st.title("🧠 Brain Tumor Detection")
st.caption("AI-powered brain MRI analysis using YOLOv8")

if not MODEL_PATH.exists():
    st.error("best.pt was not found. Put best.pt in the same folder as app.py.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error("The trained model could not be loaded.")
    st.exception(e)
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded MRI")
    st.image(image, use_container_width=True)

    confidence = st.slider(
        "Minimum confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )

    if st.button("🔍 Detect Tumor", use_container_width=True):
        with st.spinner("Analyzing MRI..."):
            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=640,
                verbose=False,
            )

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            st.warning("No detection was found above the selected confidence threshold.")
        else:
            st.success(f"Detection completed — {len(boxes)} detection(s) found.")

            annotated = result.plot()
            annotated_image = Image.fromarray(annotated[:, :, ::-1])

            st.subheader("Detection Result")
            st.image(annotated_image, caption="Annotated MRI", use_container_width=True)

            st.subheader("Detected Classes")
            names = result.names

            for i, box in enumerate(boxes):
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                st.write(f"**{i + 1}. {class_name}** — Confidence: **{score * 100:.2f}%**")

st.divider()
st.caption(
    "Educational/research demonstration only. This application is not a medical "
    "diagnostic tool and should not be used for clinical decisions."
)
