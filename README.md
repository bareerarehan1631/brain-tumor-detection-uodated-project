# 🧠 Brain Tumor Detection — Streamlit

A Streamlit web application for demonstrating a YOLOv8 brain MRI tumor-detection model.

## Project structure

```text
brain-tumor-streamlit/
├── app.py
├── best.pt
├── requirements.txt
└── README.md
```

## Model

`best.pt` is the trained YOLO model supplied with this project. The application loads the class names directly from the trained model, so class labels do not need to be hard-coded in `app.py`.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload **app.py**, **best.pt**, **requirements.txt**, and **README.md**.
3. Open Streamlit Community Cloud and choose **Create app**.
4. Select your GitHub repository and branch.
5. Set the main file to `app.py`.
6. Click **Deploy**.

### Important

The model file must be named exactly `best.pt` and must be in the same repository folder as `app.py`.

## How it works

1. Upload a brain MRI image.
2. The app loads the trained YOLO model from `best.pt`.
3. The image is passed to the model.
4. Detections above the selected confidence threshold are displayed.
5. The app shows the annotated MRI and confidence scores.

## Streamlit deployment notes

The supplied model is approximately 6 MB, so it is small enough for normal GitHub repository storage. If GitHub rejects a future model because of file-size limits, use Git LFS or external model hosting.

## Medical disclaimer

This project is an educational/research demonstration. It is **not a medical diagnostic system**. Predictions should not be used as a substitute for evaluation by a qualified medical professional.
