# 👁️ Enhancing Trust in Automated Screening: A Swin Transformer and CNN Hybrid Network for Diabetic Retinopathy Diagnosis with Natural Language Explanation

An advanced, explainable deep learning framework that leverages a hybrid architecture combining global contextual awareness with local feature extraction to accurately grade Diabetic Retinopathy (DR) from retinal fundus images.

## 📖 How to View the Source Code

> [!IMPORTANT]
> Because this Jupyter Notebook (`.ipynb`) contains rich training history visualizations, interactive loss/accuracy curves, and high-resolution medical imaging layers, the file size exceeds GitHub's standard static web-rendering limits. 

To view the fully executed pipeline, interactive dashboards, and complete source code without rendering freezes or truncation errors, please **click the orange `Open in nbviewer` badge** below or click the direct quick link:

[![Jupyter Notebook](https://img.shields.io/badge/View_Notebook-Open_in_nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/P-Gnanesh/Diabetic-Retinopathy-Detection/blob/main/ml-hl-proj-code%20%282%29.ipynb)


> 🚀 Deep Learning project for automated detection of diabetic retinopathy from retinal fundus images with explainable AI insights.

---

## 📌 Problem Statement
Diabetic Retinopathy (DR) is a widespread microvascular complication of diabetes that requires timely and accurate diagnosis to prevent irreversible blindness. However, manual screening by ophthalmologists is resource-intensive, time-consuming, and prone to inter-observer variability, creating a massive demand for scalable, automated diagnostic solutions.  Despite recent progress in deep learning, existing automated screening systems suffer from three critical limitations that prevent their deployment in real-world clinical environments:Architectural Limitations: Pure Convolutional Neural Networks (CNNs) are constrained by local receptive fields, failing to capture long-range spatial dependencies like peripheral neovascularization. Conversely, pure Vision Transformers (ViTs) model global context effectively but often miss fine-grained, localized lesions such as microaneurysms that are essential for early-stage detection.  The "Black Box" Opaque Boundary: Existing highly accurate models operate as black boxes, outputting classification grades without interpretable diagnostic reasoning, which fundamentally undermines clinical trust and regulatory approval.  Cognitive Interpretation Burden: Current Explainable AI (XAI) approaches rely heavily on visual heatmaps (e.g., Grad-CAM), which place a heavy cognitive burden on clinicians to mentally map highlighted regions to actual clinical terminology. There is an absolute absence of screening systems capable of translating internal attention maps into structured, natural language clinical explanations that fit directly into healthcare and teleophthalmology workflows.

---

## 🧠 Solution Approach
## ⚙️ System Methodology & Proposed Solution

To overcome the diagnostic limitations of standalone architectures, the proposed framework introduces a multi-modal, dual-engine hybrid architecture: the **Swin Transformer and CNN Hybrid Network with an integrated Natural Language Generation (NLG) module**. The overall solution pipeline operates across four precise, sequential engineering layers to transform a raw retinal image into an accurate multi-class classification and a transparent textual clinical description.

---

### 🎨 Layer 1: Artifact-Resilient Image Preprocessing
Before entering the feature extraction networks, raw retinal fundus photographs undergo a standardized mathematical preprocessing sequence to eliminate image noise, remove lighting artifacts, and highlight hidden lesions:

* **🖼️ Aspect-Ratio Preserving Normalization:** Input images are resized to a uniform architecture-compatible resolution of $224 \times 224$ pixels. This creates data consistency across the network while preserving the true spherical geometry of the fundus structures.
* **⚡ Ben Graham Contrast Enhancement:** To enhance the visibility of subtle vascular abnormalities against background tissue, the pipeline applies a Gaussian blur color-subtraction matrix:
  $$I_{enhanced} = 4I - 4G_{\sigma}(I) + 128$$
  This formula isolates high-frequency spatial details, making microaneurysms, hemorrhages, and exudates drastically sharper for the model branches to pinpoint.
* **🎯 Circular Retinal Masking:** A spatial filtering mask calculates the exact circular boundaries of the active retina, clipping out black background margins and edge noise to ensure the extractors process only real anatomical regions.

---

### 🔄 Layer 2: Asymmetric Dual-Stream Feature Extraction
Rather than relying on a single model backbone, the core of the solution splits feature extraction into two parallel, highly specialized branches to simultaneously capture micro and macro structures:

#### 🧬 Stream A: The CNN Local-Texture Engine (EfficientNet-B0)
* **🎯 Purpose:** Targeted extraction of micro-lesions.
* **⚙️ Mechanism:** Using small, localized receptive fields and standard convolution operations, this branch extracts high-frequency, fine-grained visual features. It is heavily optimized to preserve the distinct structural boundaries of tiny microaneurysms, hard exudates, and localized blood leakages.
* **🔌 Channel Attention:** This stream is further refined using Squeeze-and-Excitation (SE) blocks to compute global channel-wise statistics and adaptively recalibrate feature weights, ensuring the most discriminative lesion channels are prioritized.

#### 🌐 Stream B: The Swin Transformer Global-Context Engine
* **🎯 Purpose:** Capturing macro-structural alterations and long-range structural dependencies.
* **⚙️ Mechanism:** The Swin Transformer branch solves the high computational complexity of traditional transformers by computing self-attention within localized, non-overlapping windows:
  $$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right)V$$
* **🪟 Shifted Windowing:** By shifting the window partition between consecutive layers, it allows features to interact across boundaries. This lets the network model long-range spatial patterns, such as extensive peripheral neovascularization networks, which pure CNNs miss due to their restricted local receptive fields.

---

### 🔀 Layer 3: Multi-Head Cross-Attention Fusion
The true bottleneck of hybrid networks is the fusion of asymmetric data shapes. Instead of using simple concatenation or pooling—which destroys spatial relationships—this solution introduces a **Multi-Head Cross-Attention (MHCA) Layer**.

* **🔗 The Integration:** The global contextual embeddings from the Swin Transformer stream are projected into a Query ($Q$) matrix, while the dense spatial lesion features from the CNN stream are projected into Key ($K$) and Value ($V$) matrices.
* **🧠 Dynamic Querying:** By computing cross-attention, the system allows the global background context to dynamically query local lesion texture spaces. For example, if the global branch registers generalized background retinopathy patterns, the cross-attention layer automatically focuses high-weight vectors onto the precise local keys where micro-lesions reside.
* **📊 Classification Output:** The unified cross-attention feature vector is passed to a fully connected dense layer using GELU activation and dropout regularization. To address severe data distribution inequalities, the model employs a class-weighted cross-entropy loss function paired with label smoothing to output final class probabilities across the 5-grade clinical severity spectrum.

---

### 📝 Layer 4: Explainable AI (XAI) Natural Language Generation
To move past the traditional "black box" limitations of deep learning in medicine, the final component of the solution maps internal network attention directly to a **Natural Language Generation (NLG) module**.

* **🔤 Visual-to-Textual Translation:** Instead of generating purely visual heatmaps (like Grad-CAM) that leave clinicians with the cognitive burden of interpreting colored regions, this module maps the cross-attention feature weights directly into text-generation clinical explanations.
* **📋 Automated Clinical Reporting:** The network translates the active, high-weight attention regions into clinically coherent text, automatically generating descriptions of the detected lesions and providing a transparent, written rationale for the assigned severity grade. This allows the output to integrate seamlessly into electronic health records and real-world teleophthalmology screening workflows.

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
