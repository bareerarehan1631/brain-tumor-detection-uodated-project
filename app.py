import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Model Path
# -----------------------------
MODEL_PATH = Path(__file__).parent / "best.pt"


# -----------------------------
# Load YOLO Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# -----------------------------
# Title
# -----------------------------
st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain MRI image to detect a possible tumor using YOLOv8.")


# -----------------------------
# Check Model
# -----------------------------
if not MODEL_PATH.exists():
    st.error("❌ best.pt was not found.")
    st.info("Make sure best.pt is in the same folder as app.py.")
    st.stop()


# -----------------------------
# Load Model
# -----------------------------
try:
    model = load_model()
except Exception as e:
    st.error("❌ The trained YOLO model could not be loaded.")
    st.exception(e)
    st.stop()


# -----------------------------
# Upload MRI
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded MRI")

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )

    # -----------------------------
    # Confidence Slider
    # -----------------------------
    confidence = st.slider(
        "Detection Confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.40,
        step=0.05
    )

    # -----------------------------
    # Detection Button
    # -----------------------------
    if st.button(
        "🔍 Detect Tumor",
        use_container_width=True
    ):

        with st.spinner("Analyzing MRI..."):

            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=640,
                verbose=False
            )

        result = results[0]
        boxes = result.boxes

        # -----------------------------
        # No Detection
        # -----------------------------
        if boxes is None or len(boxes) == 0:

            st.warning(
                "No tumor detection was found above the selected confidence threshold."
            )

        # -----------------------------
        # Detection Found
        # -----------------------------
        else:

            st.success(
                f"Detection completed — {len(boxes)} detection(s) found."
            )

            # -----------------------------
            # Annotated Image
            # -----------------------------
            annotated = result.plot()

            # YOLO returns BGR, convert to RGB
            annotated_image = Image.fromarray(
                annotated[:, :, ::-1]
            )

            st.subheader("Detection Result")

            st.image(
                annotated_image,
                caption="Annotated MRI",
                use_container_width=True
            )

            # -----------------------------
            # Detection Details
            # -----------------------------
            st.subheader("Detected Tumor Details")

            names = result.names

            for i, box in enumerate(boxes):

                class_id = int(
                    box.cls[0].item()
                )

                score = float(
                    box.conf[0].item()
                )

                if isinstance(names, dict):
                    class_name = names.get(
                        class_id,
                        str(class_id)
                    )
                else:
                    class_name = str(class_id)

                st.write(
                    f"**{i + 1}. {class_name}** "
                    f"— Confidence: **{score * 100:.2f}%**"
                )


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Educational/research demonstration only. "
    "This application is not a medical diagnostic tool "
    "and should not be used for clinical decisions."
)
