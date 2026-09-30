# 🩻 Medical Image Classification — AI Prototype

A CNN-based image classification prototype built with **TensorFlow/Keras** and **Streamlit**. The project demonstrates an end-to-end image classification workflow using synthetic normal/abnormal image patterns.

> ⚠️ **Important:** This is a research/demo prototype only. It is **not clinically validated** and must not be used for medical diagnosis or clinical decision-making.

---

## 📌 Project Overview

The objective of this project is to demonstrate a basic deep-learning image classification pipeline:

- Generate or load image data
- Preprocess images
- Train a Convolutional Neural Network (CNN)
- Evaluate the model
- Save the trained model
- Build an interactive Streamlit application
- Display class predictions and abnormal-class probability

The prototype supports two classes:

- `normal`
- `abnormal`

---

## 📊 Dataset

The training script supports two dataset modes.

### 1. Real Dataset Mode

If the following folders exist:

```text
data/
├── normal/
└── abnormal/
```
# 🧠 Model Architecture

The prototype uses a Convolutional Neural Network implemented using TensorFlow/Keras.

Main components include:

Convolutional layers
ReLU activation
Global Average Pooling
Dropout
Dense layer
Sigmoid output layer for binary classification

Input image size:

128 × 128 × 3

Output:

Normal / Abnormal
📈 Model Evaluation

The training pipeline records:

Training accuracy
Validation accuracy
AUC
Validation loss

Because the default dataset is synthetic and based on generated visual patterns, the resulting metrics only demonstrate that the software pipeline can learn the generated patterns.

They do not represent medical or clinical accuracy.

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application.

The application allows the user to:

Upload a PNG/JPG/JPEG image
Preview the uploaded image
Run the trained CNN model
View abnormal-class probability
View the predicted class

The application also clearly warns users that the model is a research/demo prototype.

# 🧪 Demo Results
Normal Synthetic Image

The model is tested using a synthetic image generated from the same type of patterns used during training.

Abnormal Synthetic Image

The model is also tested using a synthetic abnormal-pattern image.

Out-of-Distribution Example

A real-world image such as a tablet photograph can also be uploaded to demonstrate an important limitation.

The model may still produce a normal or abnormal prediction because it only has two output classes and does not contain an unknown/out-of-distribution rejection class.

Therefore, a prediction such as 100% abnormal for an unrelated real-world image should not be interpreted as a medical finding.

# 📁 Project Structure
03_medical_image_ai/
│
├── artifacts/
│   ├── medical_classifier.keras
│   └── training_accuracy.png
│
├── demo_images/
│   ├── demo_normal.png
│   └── demo_abnormal.png
│
├── screenshots/
│   ├── normal_prediction.png
│   ├── abnormal_prediction.png
│   └── out_of_distribution_limitation.png
│
├── app.py
├── train.py
├── make_demo_images.py
├── requirements.txt
├── README.md
└── .gitignore

---
## ⚙️ Installation

Create and activate a Python environment, then install the dependencies:

pip install -r requirements.txt
▶️ Training

Run:

python train.py

The trained model will be saved under:

artifacts/medical_classifier.keras
🚀 Run the Streamlit App

Run:

streamlit run app.py

The application will open in the browser.

# 🧪 Generate Demo Images

To generate the synthetic images used for demonstration:

python make_demo_images.py

This creates:

demo_images/
├── demo_normal.png
└── demo_abnormal.png

These images are intended for demonstrating the behavior of the prototype.

# ⚠️ Limitations

This project is an internship-level AI prototype and has several limitations:

The default dataset is synthetic.
The model is not clinically validated.
Only two classes are supported.
There is no dedicated unknown/out-of-distribution class.
Predictions on unrelated real-world images are not medically meaningful.
No DICOM/clinical imaging pipeline is implemented.
No clinical metadata is used.
No medical image segmentation is implemented.
No clinical deployment or patient-data workflow is included.
Performance on synthetic data cannot be generalized to real patients or clinical environments.
# 🔮 Future Improvements

Possible future improvements include:

Use a permitted real medical imaging dataset.
Support DICOM medical images.
Add multi-class or multi-label classification.
Add out-of-distribution detection.
Add Grad-CAM or other explainability techniques.
Perform more extensive validation.
Add FastAPI for model serving.
Containerize the application with Docker.
Deploy using cloud infrastructure.
Add monitoring and model-drift detection.
Retrain and validate the model using appropriately licensed real-world data.
# 🛡️ Disclaimer

This project is for educational, research, and software-demonstration purposes only.

It is not a medical diagnostic system and must not be used to diagnose, treat, or make clinical decisions about any medical condition.

The default synthetic dataset and its evaluation results should not be interpreted as evidence of clinical effectiveness.

# 📌 Project Status

Completed — Internship Prototype

The project demonstrates an end-to-end CNN image-classification workflow with TensorFlow/Keras and a Streamlit interface.


### 🔥 One important thing

Your original README says:

> “Medical Image Diagnosis”

I'd change that to **“Medical Image Classification — AI Prototype”**.

That's much safer and more accurate because we're **classifying synthetic patterns**, not actually diagnosing a medical condition.

And your 3 screenshots fit perfectly into this README:

```text
normal_prediction.png
abnormal_prediction.png
out_of_distribution_limitation.png
```