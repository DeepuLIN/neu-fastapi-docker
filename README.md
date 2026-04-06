# 🚀 NEU Surface Defect Classification API (FastAPI + Docker)

A production-ready machine learning API for surface defect classification using deep learning, deployed with FastAPI and Docker.

---

## 📌 Overview

This project provides an end-to-end pipeline for:

* Training a deep learning model for surface defect classification (NEU dataset)
* Serving predictions via a FastAPI REST API
* Containerizing the application using Docker for reproducibility and deployment

---

## 🧠 Features

* ✅ Deep Learning model (PyTorch / PyTorch Lightning)
* ✅ Image classification API with FastAPI
* ✅ Grad-CAM visualization for model explainability
* ✅ Dockerized deployment (portable & reproducible)
* ✅ Clean modular project structure

---

## 📂 Project Structure

```
.
├── api/                # FastAPI application
├── src/                # Training, inference, utilities
├── models/             # Saved checkpoints (ignored in Git)
├── data/               # Dataset (ignored in Git)
├── tests/              # Sample test images
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup (Local)

### 1️⃣ Clone the repository

```
git clone https://github.com/DeepuLIN/neu-fastapi-docker.git
cd neu-fastapi-docker
```

---

### 2️⃣ Create environment

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 3️⃣ Run FastAPI

```
uvicorn api.main:app --reload
```

Open:

```
http://localhost:8000/docs
```

---

## 🐳 Run with Docker

### 1️⃣ Build image

```
docker build -t neu-api .
```

---

### 2️⃣ Run container

```
docker run -p 8000:8000 neu-api
```

---

### 3️⃣ Access API

```
http://localhost:8000/docs
```

---

## 📡 API Endpoints

### 🔹 Health Check

```
GET /
```

---

### 🔹 Image Prediction

```
POST /predict/image
```

Upload an image file and receive:

* Predicted class
* Confidence score
* Grad-CAM visualization (if enabled)

---

## 🧪 Example Usage

Using curl:

```
curl -X POST "http://localhost:8000/predict/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/1.jpg"
```

---

## 📊 Model

* Architecture: ResNet / MobileNet (PyTorch)
* Framework: PyTorch Lightning
* Explainability: Grad-CAM
* Experiment tracking: MLflow (optional)

---

## ⚠️ Notes

* Dataset is **not included** in this repository
* Model checkpoints are ignored for lightweight version control
* Use `src/download_data.py` or external sources to fetch dataset

---

## 🚀 Future Improvements

* CI/CD pipeline (GitHub Actions)
* Model versioning with MLflow
* Deployment to cloud (AWS / Azure / GCP)
* Batch inference support

---

## 👤 Author

Deepak L
Machine Learning Engineer | Computer Vision | MLOps

---

## ⭐ If you found this useful

Give it a star ⭐ on GitHub — it helps a lot!
