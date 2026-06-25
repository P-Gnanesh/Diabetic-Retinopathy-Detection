# 🧠 Diabetic Retinopathy Detection using Swin CNN
[![Jupyter Notebook](https://img.shields.io/badge/View_Notebook-Open_in_nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/P-Gnanesh/Diabetic-Retinopathy-Detection/blob/main/ml-hl-proj-code%20%282%29.ipynb)

> 🚀 Deep Learning project for automated detection of diabetic retinopathy from retinal fundus images with explainable AI insights.

---

## 📌 Problem Statement
Diabetic Retinopathy is a leading cause of blindness if not detected early.  
This project builds a robust deep learning model to classify retinal images and assist in early diagnosis.

---

## 🧠 Solution Approach
- 🔍 Data preprocessing & augmentation
- 🧩 Feature understanding using PCA
- 🤖 Swin CNN model for classification
- 🔥 Grad-CAM for model interpretability
- 📊 Performance evaluation using multiple metrics

---

## 📊 Model Performance
| Metric        | Score   |
|--------------|--------|
| ✅ Accuracy   | **86.36%** |
| 📈 AUC-ROC    | **0.9421** |
| ⚖️ F1 Score   | **76.78%** |

---

## 📈 Visual Insights
- 📊 Confusion Matrix
- 📉 ROC Curve & PR Curve
- 🧬 Feature Correlation Heatmap
- 🧠 Grad-CAM Visualizations
- 📦 Training History Graphs

---

## 🧰 Tech Stack
- Python 🐍
- NumPy, Pandas  
- Scikit-learn  
- OpenCV  
- Matplotlib & Seaborn  
- Albumentations  
- TIMM (PyTorch models)

---

## 📁 Dataset
📌 APTOS 2019 Blindness Detection  
🔗 https://www.kaggle.com/c/aptos2019-blindness-detection  

> ⚠️ Dataset not included due to size constraints.

---

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/P-Gnanesh/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook
