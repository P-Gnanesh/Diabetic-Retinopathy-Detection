# 🧠 Diabetic Retinopathy Detection using Swin CNN
[![Jupyter Notebook](https://img.shields.io/badge/View_Notebook-Open_in_nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/P-Gnanesh/Diabetic-Retinopathy-Detection/blob/main/ml-hl-proj-code%20%282%29.ipynb)

> 🚀 Deep Learning project for automated detection of diabetic retinopathy from retinal fundus images with explainable AI insights.

---

## 📌 Problem Statement
Diabetic Retinopathy (DR) is a widespread microvascular complication of diabetes that requires timely and accurate diagnosis to prevent irreversible blindness. However, manual screening by ophthalmologists is resource-intensive, time-consuming, and prone to inter-observer variability, creating a massive demand for scalable, automated diagnostic solutions.  Despite recent progress in deep learning, existing automated screening systems suffer from three critical limitations that prevent their deployment in real-world clinical environments:Architectural Limitations: Pure Convolutional Neural Networks (CNNs) are constrained by local receptive fields, failing to capture long-range spatial dependencies like peripheral neovascularization. Conversely, pure Vision Transformers (ViTs) model global context effectively but often miss fine-grained, localized lesions such as microaneurysms that are essential for early-stage detection.  The "Black Box" Opaque Boundary: Existing highly accurate models operate as black boxes, outputting classification grades without interpretable diagnostic reasoning, which fundamentally undermines clinical trust and regulatory approval.  Cognitive Interpretation Burden: Current Explainable AI (XAI) approaches rely heavily on visual heatmaps (e.g., Grad-CAM), which place a heavy cognitive burden on clinicians to mentally map highlighted regions to actual clinical terminology. There is an absolute absence of screening systems capable of translating internal attention maps into structured, natural language clinical explanations that fit directly into healthcare and teleophthalmology workflows.

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
