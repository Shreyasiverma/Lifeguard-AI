# Lifeguard AI: Health prediction and diagnosis system

## 📌 Overview

**Lifeguard AI** is an intelligent, Machine Learning-driven diagnostic support system. By analyzing symptom clusters, it provides rapid disease prediction, risk stratification, and actionable next steps for clinical intervention.

Our mission is to bridge the gap between "noticing a symptom" and "getting professional care" through data-driven transparency.

---

## 🚀 Enhanced Features

* **Intelligent Prediction Engine**: Analyzes 5-symptom clusters using high-accuracy ML models.
* **Confidence Scoring**: Displays a probability percentage for each prediction to ensure transparency.
* **Urgency & Triage**: Categorizes conditions into *Low*, *Medium*, or *High* urgency with specific action protocols.
* **Clinical Reasoning**: Provides a breakdown of *why* the model reached its conclusion, linking symptoms to the predicted pathology.
* **Care Navigation**:
* **Specialist Finder**: Automatically recommends the correct type of doctor (e.g., Cardiologist for chest pain).
* **Geo-Location Services**: Integrated mapping to find the nearest hospitals and clinics.
* **Emergency SOS**: A one-tap system for immediate emergency service contact.


* **Safety First**: A persistent, high-visibility Warning Banner and localized precautionary advice for every diagnosis.

---

## 🛠️ Modern Tech Stack

* **Core**: Python, Scikit-learn, NumPy, Pandas.
* **Frontend**: Streamlit (Enhanced UI with custom CSS).
* **APIs**: Google Maps API (Hospital Search), Twilio (Emergency Alerts).

---

## 📊 The Intelligence Workflow

| Step | Process | Description |
| --- | --- | --- |
| **1** | **Symptom Input** | User selects 5 symptoms via an intuitive multi-select UI. |
| **2** | **ML Inference** | The model calculates the most probable condition and a confidence score. |
| **3** | **Risk Analysis** | The system assesses urgency and generates a "Reasoning Report". |
| **4** | **Actionable Care** | UI displays nearby specialists, hospitals, and emergency call buttons. |

---

## ▶️ Setup & Installation

### 1️⃣ Clone & Navigate

```bash
git clone https://github.com/Shreyasiverma/techtitansYUKTI


```

### 2️⃣ Environment Setup

```bash
pip install -r requirements.txt

```

### 3️⃣ Launch the Hub

```bash
streamlit run app.py

```

---

## 🚨 Safety & Clinical Disclaimer

> **CRITICAL**: Lifeguard AI is a preliminary screening tool for educational and hackathon purposes. It is not a replacement for professional medical diagnosis, advice, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
