# 🧠 Brain Tumor Detection

An AI-powered **Brain Tumor Detection** application built using **YOLOv8** and **Streamlit**.

The application allows users to upload a brain MRI image and uses a trained YOLOv8 object detection model to identify possible tumor regions.

## 🚀 Live Application

The application is deployed using Streamlit Community Cloud.

## 📌 Project Overview

This project uses a YOLOv8 model trained on a brain tumor dataset obtained from Roboflow.

The trained model can:

* Accept brain MRI images as input
* Detect possible tumor regions
* Display the detected regions on the MRI
* Show the detected class
* Display the confidence score
* Allow users to adjust the detection confidence threshold

## 🤖 Model Information

**Model:** YOLOv8 Nano (`yolov8n.pt`)
**Task:** Object Detection
**Training Epochs:** 50
**Image Size:** 640 × 640
**Batch Size:** 8
**Dataset:** Brain Tumor Dataset from Roboflow

The trained model was saved as:

```text
runs/detect/brain_tumor_detection_yolov8_roboflow/weights/best.pt
```

The final `best.pt` file is used by the Streamlit application for inference.

## 📂 Project Structure

```text
brain-tumor-detection/
│
├── app.py
├── best.pt
├── requirements.txt
└── README.md
```

### `app.py`

Contains the Streamlit application and YOLOv8 inference code.

### `best.pt`

Contains the trained YOLOv8 model weights.

### `requirements.txt`

Contains the Python libraries required to run the application.

### `README.md`

Contains project documentation and deployment information.

## 🛠️ Technologies Used

* Python
* YOLOv8
* Ultralytics
* Streamlit
* OpenCV
* Pillow
* Roboflow
* Google Colab

## ▶️ Run Locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Streamlit Deployment

To deploy this project using Streamlit Community Cloud:

1. Upload the following files to your GitHub repository:

```text
app.py
best.pt
requirements.txt
README.md
```

2. Connect the GitHub repository to Streamlit Community Cloud.

3. Select:

```text
Main file: app.py
```

4. Deploy the application.

Streamlit will automatically install the dependencies from `requirements.txt`.

## 🔍 How to Use

1. Open the application.
2. Upload a brain MRI image.
3. Select the desired detection confidence.
4. Click **Detect Tumor**.
5. View the annotated MRI.
6. Review the detected class and confidence score.

## ⚠️ Disclaimer

This project is developed for **educational and research demonstration purposes only**.

It is **not a medical diagnostic tool** and should not be used for clinical diagnosis, treatment decisions, or other medical decisions.

Always consult a qualified medical professional for medical evaluation.

## 👩‍💻 Project

**Brain Tumor Detection using YOLOv8**

Developed as a machine learning / deep learning project demonstrating object detection and Streamlit deployment.

