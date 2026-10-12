# 🔍 Omni-Detect V6: Deepfake Detection

### Three-Branch Gated Fusion Architecture for Deepfake Detection

## 📌 Overview
**Omni-Detect V6** is a deep learning-based system designed to detect deepfake images and videos. It uses a Three-Branch Gated Fusion Architecture to analyse spatial artifacts, frequency patterns, and temporal inconsistencies. The system combines SRM filters, ConvNeXt-Tiny, CBAM attention, and BiGRU for feature extraction and analysis. A Gated Fusion Network combines these features to classify media as Real or Deepfake. The project also provides a user-friendly Gradio web interface for media verification.

## 🎯 Key Features
- **Spatial Analysis:** SRM filters, ConvNeXt-Tiny, and CBAM attention.
- **Frequency Analysis:** FFT and DCT for detecting spectral artifacts.
- **Temporal Analysis:** BiGRU for identifying inconsistencies across video frames.
- **Gated Fusion:** Dynamically combines features from all three branches.
- **Web Interface:** Gradio-based interface for image and video analysis.

## 🛠️ Technologies Used
Python, PyTorch, OpenCV, ConvNeXt-Tiny, CBAM, BiGRU, RetinaFace, Gradio, FFT, and DCT.

## 📊  Results

| Evaluation Metric | Performance |
| ----------------- | ----------- |
| Accuracy          | 87.8%       |
| Precision         | 97.1%       |
| Recall            | 83.2%       |

## 📂 Dataset
**FaceForensics++** — a dataset containing authentic and manipulated facial videos.

[Dataset Repository](https://github.com/ondyari/FaceForensics)

## 🌍 Applications
- Digital forensics
- Social media content verification
- Fake video detection
- Cybersecurity and misinformation prevention

## 📚 References
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [PyTorch](https://pytorch.org/)
- [Gradio](https://www.gradio.app/)
- [InsightFace](https://github.com/deepinsight/insightface)

