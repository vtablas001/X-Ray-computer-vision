# AI-Powered Chest X-Ray Diagnosis for El Salvador's Public Health System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Prototype%20Demo-green)]()


## 🏥 Project Overview
This project leverages convolutional neural networks to power computer vision and artificial intelligence to support early, automated detection of respiratory diseases through chest X-Ray analysis. 

Developed as a proof-of-concept for the **Ministry of Health (MINSAL)** and the **Teachers' Wellbeing Institute (ISBM)**, this system demonstrates how AI can empower medical professionals with tools for faster and more accurate diagnosis. It is aligned with UNDP’s 2025 AI for Sustainable Development Atlas, aiming to improve healthcare access in resource-constrained settings.

### 🔗 Live Demo
[Platform Demo](https://v0-pulmonary-ai-demo.vercel.app/)

---

## ⚠️ Medical Disclaimer
> **This tool is intended for research and development purposes only.** It is designed to assist, not replace, medical professionals. It is not a certified medical device and should not be used as the sole basis for diagnosis or treatment.

---

## 🎯 Context & Objectives
Respiratory diseases remain a major public health challenge in El Salvador. This initiative is designed to:
1.  **Reduce diagnostic delays** in high-demand units.
2.  **Alleviate the burden on radiologists** by automating the triage of normal vs. pathological scans.
3.  **Improve healthcare access** in rural or underserved areas where specialists are scarce.

## 🧠 Technical Approach & Data Scale

### 📂 The Dataset: 10,000+ Images
To ensure clinical relevance and statistical significance, the model was trained and validated on a robust dataset of **over 10,000 chest X-ray images**. 
* **Scale:** 10,000+ processed images.
* **Diversity:** Includes a balanced mix of "Normal," "Viral Pneumonia," and "COVID-19" cases to prevent model bias.
* **Preprocessing:** All images underwent standardization (resizing, CLAHE contrast enhancement) to simulate real-world input variability.

### ⚙️ The Model: VGG16 & Transfer Learning
* **Architecture:** We employ a pre-trained **VGG16 Convolutional Neural Network (CNN)**.
* **Transfer Learning:** Weights from ImageNet were used as a starting point, followed by fine-tuning specific layers to recognize radiographic features (opacities, consolidations) rather than generic objects.
* **Data Augmentation:** Techniques such as rotation, zoom, and brightness shifts were applied during training to improve the model's ability to generalize to new, unseen X-rays.

### 🔎 Target Conditions
The system classifies images into three categories:
* **Normal** (Healthy)
* **Viral Pneumonia**
* **COVID-19**

---

## 📓 Interactive Analysis (Jupyter Notebook)
The core logic of this project is documented in the included Jupyter Notebook. This file serves as the technical proof-of-concept, providing a transparent walkthrough of:
* **Exploratory Data Analysis (EDA):** Visualization of the 10,000+ image dataset distribution.
* **Training Pipeline:** Real-time training logs showing accuracy/loss convergence.
* **Performance Metrics:** Confusion matrices and classification reports on the test set.

[**📄 View the Source Notebook**](Torax_X_Ray_pneumonia_CNN.ipynb)

---

## 🛠️ Language & Core Tools

### Primary Language
* **Python 3.10+**: Chosen for its dominance in the data science and medical AI ecosystem.

### Key Libraries & Frameworks
* **Deep Learning:** `TensorFlow`, `Keras` (Model architecture and training).
* **Computer Vision:** `OpenCV` (`cv2`), `Pillow` (Image preprocessing and augmentation).
* **Data Manipulation:** `NumPy`, `Pandas` (Tensor handling and dataset management).
* **Visualization:** `Matplotlib`, `Seaborn` (Heatmaps, confusion matrices, and training curves).
* **Deployment:** `FastAPI` (Backend inference), `Vercel` (Frontend hosting).

---

## 🚀 Usage

To run the analysis locally using the provided notebook:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vtablas001/xray-computer-vision.git
