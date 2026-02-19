🩺 Analysis and Evaluation of Vitiligo Severity
> **Semantic Segmentation in Clinical Dermatology**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

## 🖼️ Application Preview

<p align="center">
  <img src="https://github.com/user-attachments/assets/1ae5bfca-27ce-465f-8c79-67ca20d44daa" width="600" alt="Vitiligo Segmentation Analysis Output">
  <br>
  <i>Figure 1: Sample output showing original skin image, predicted segmentation mask, and calculated area ratio.</i>
</p>

This project is a Flask-based web application for vitiligo lesion segmentation using a **SAM (Segment Anything Model) + U-Net** hybrid deep learning architecture. The system focuses on **model inference, web serving, and deployability** rather than model training.

---

## 📝 Project Overview

This project presents a **deep learning–based vitiligo segmentation system** that combines the **Segment Anything Model (SAM)** with a **U-Net** architecture to identify and quantify vitiligo-affected regions in skin images.

The system includes an end-to-end inference pipeline and a lightweight Flask-based web interface for visualization and analysis. It is specifically designed to focus on **model inference, web serving, and deployability** in clinical dermatology contexts.

---

## ✨ Key Features

* 🖼️ **Advanced Segmentation:** Automated vitiligo lesion detection via image upload.
* 📊 **Quantitative Analysis:** Automatic calculation of the lesion area ratio.
* 🚀 **Performance Optimized:** Support for batch inference and latency benchmarking.
* 🐳 **Containerized:** Docker-ready for consistent deployment across environments.
* 💻 **User Interface:** Clean web visualization powered by Flask and Jinja2 templates.

---


## 🏗️ System Architecture

| Component | Detail |
| :--- | :--- |
| **Backbone** | U-Net |
| **Encoder Guidance** | Segment Anything Model (SAM) |
| **Output** | Binary Segmentation Mask |
| **Post-processing** | Lesion Area Ratio Calculation |

> [!IMPORTANT]
> Due to academic publication constraints, pretrained weights (`.pth`) and training notebooks are excluded. The system automatically falls back to **CPU** if CUDA is unavailable.

---

## 🔬 Dataset

The dataset consists of dermatological images collected from multiple academic and open-access sources. 

Due to ethical considerations, patient privacy, and ongoing academic publication, the dataset is not publicly available. Researchers interested in academic collaboration may contact the author.

---

## 📂 Project Structure

```text
VITILIGO_APP/
├── 🧠 model/               # Model architecture & segmentation logic
│   ├── sam_unet.py
│   ├── vitiligosegmenter.py
│   └── segment_anything/
├── 🌐 templates/           # Web UI (HTML)
├── 🎨 static/              # CSS & Static assets
├── 🧪 test_images/         # Sample test data
├── 🐳 Dockerfile           # Container configuration
├── 📄 app.py               # Main Flask application
└── 📊 latency_log.csv      # Inference latency logs
```


🚀 Run with Docker
To get the application up and running:

1. Build Image
```text
docker build -t vitiligo-app .
```
2. Run Container
```bash
docker run -p 8080:8080 vitiligo-app
```
3. Access
```text
Open: http://localhost:8080
```

## 📌 Notes
* **Model Weights:** Pretrained weights (.pth) are not included due to ongoing academic publication.

* **Hardware:** The application automatically falls back to CPU if CUDA is not available.

* **Latency:** latency_log.csv contains sample inference latency measurements.

---

## 🛠️ Technologies Used
* Language: Python

* Backend: Flask

* Deep Learning: PyTorch

* Containerization: Docker

* Frontend: HTML / CSS (Jinja Templates)

---

👤 Author
Ebrar Türüdü 

> [!IMPORTANT]  
> **Disclaimer:** This tool is developed for research purposes and is not intended for clinical diagnosis.










