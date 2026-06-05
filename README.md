# 🚀 Luminate - Loan Prediction Backend API

Welcome to the backend service of **Luminate**, an automated credit risk assessment and loan prediction platform. This repository houses the production-ready machine learning pipeline and API endpoints that power the Luminate intelligent decision engine.

---

## 🧠 Machine Learning & Architecture
The system utilizes a trained **K-Nearest Neighbors (KNN)** classification model to predict loan eligibility based on user financial profiles. 

* **FastAPI Framework:** Handles asynchronous API routing with high performance and low latency.
* **Data Scaling:** Features automated inputs normalization via a pre-trained robust Scaler pipeline.
* **Decoupled Design:** Built using a modern detached frontend/backend architecture for seamless scalability.

---

## 📂 Repository Structure
The repository is optimized for minimal deployment footprint:
```text
├── assets/
│   ├── model.joblib      # Trained KNN Classification Model
│   └── scaler.joblib     # Pre-trained Data Scaler
├── main.py               # Core FastAPI Application & Prediction Routes
├── requirements.txt      # Production Dependencies
└── README.md             # Project Documentation
