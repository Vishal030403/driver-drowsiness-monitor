# 🚗 Driver Drowsiness Detection System

A complete **AI-powered web application** designed to detect driver drowsiness in real time using **Computer Vision** and a **Convolutional Neural Network (CNN)** model.
The system integrates a Python backend (FastAPI), PostgreSQL database, and a responsive frontend built with HTML, CSS, and JavaScript.

---

## 📘 Overview

Driver fatigue is a major cause of road accidents.
This project aims to **detect drowsiness through facial features** and trigger an alert if a driver shows signs of sleepiness.

The model is trained on a **3GB dataset of human faces**, labeled as **Drowsy** and **Alert**, and deployed as part of a full-stack web application.

---

## 🧠 Features

* 🧍‍♂️ Real-time video stream for driver monitoring
* 🤖 CNN-based deep learning model for drowsiness detection
* ⚙️ FastAPI backend serving model predictions via REST API
* 🗃️ PostgreSQL database for storing logs, timestamps, and detection history
* 🌐 Clean, responsive frontend using HTML, CSS, and JavaScript
* 🎞️ Option for live camera input or local video demo mode
* ☁️ Trained using **Google Colab GPU** for high performance

---

## 🧩 Tech Stack

| Layer                    | Technology                         |
| ------------------------ | ---------------------------------- |
| **Frontend**             | HTML, CSS (Tailwind), JavaScript   |
| **Backend**              | FastAPI                            |
| **Database**             | PostgreSQL                         |
| **Machine Learning**     | TensorFlow / Keras (CNN model)     |
| **Training Environment** | Google Colab (GPU runtime)         |
| **Deployment**           | Localhost / Cloud (FastAPI server) |

---

## 🏗️ System Architecture

```
User (Browser)
    ↓
Frontend (HTML, CSS, JS)
    ↓
FastAPI Backend (Model Inference + API Routes)
    ↓
CNN Model (TensorFlow/Keras)
    ↓
PostgreSQL Database (Detection Logs & Metadata)
```

---

## 🧠 Model Training

* Dataset: ~3GB of labeled images (`Drowsy`, `Alert`)
* Model Type: Custom CNN (no transfer learning)
* Training Environment: Google Colab GPU
* Output: `drowsiness_model.h5`

### 🧾 Training Steps (Google Colab)

1. Mount Google Drive and unzip dataset
2. Train CNN using TensorFlow/Keras
3. Save model to Drive (`drowsiness_model.h5`)
4. Download trained model for local deployment
5. (Optional) Convert to TensorFlow.js for web integration

---

## ⚙️ Backend (FastAPI)

* Handles API requests for predictions
* Loads the trained `drowsiness_model.h5`
* Provides endpoints for:

  * `/predict` → Receives image/frame and returns drowsiness status
  * `/history` → Fetches detection records from PostgreSQL

Example structure:

```
backend/
 ├── main.py
 ├── model/
 │   └── drowsiness_model.h5
 ├── routes/
 │   └── detection.py
 ├── db/
 │   ├── database.py
 │   └── models.py
 └── requirements.txt
```

---

## 🗃️ Database (PostgreSQL)

Stores detection history:

* ID
* Timestamp
* Prediction (Drowsy / Alert)
* Confidence score

Database connection is managed through SQLAlchemy or asyncpg in FastAPI.

---

## 🌐 Frontend

* Responsive layout built with **Tailwind CSS**
* Integrated video feed using `<video>` tag and `navigator.mediaDevices` API
* Buttons for “Start Detection” and “Stop Detection”
* Status indicator for live monitoring
* Optional demo mode (no camera permissions required)

Example structure:

```
frontend/
 ├── index.html
 ├── style.css
 ├── script.js
 └── assets/
     └── demo.mp4
```

---

## 🚀 Running the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Vishal030403/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

### 2️⃣ Set Up Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3️⃣ Set Up Database

```bash
psql -U postgres
CREATE DATABASE drowsiness_db;
```

Configure credentials in `.env`.

### 4️⃣ Start Frontend

Open `index.html` in your browser or serve it with any local HTTP server:

```bash
python -m http.server 8000
```

---

## 🧪 API Example

**POST /predict**

```json
{
  "image": "base64-encoded-frame"
}
```

**Response**

```json
{
  "status": "Drowsy",
  "confidence": 0.91,
  "timestamp": "2025-11-03T18:25:43"
}
```

---

## 📈 Future Improvements

* Add audio alerts or vibration notifications
* Deploy model on cloud (AWS / Render / Railway)
* Improve CNN accuracy with more facial landmarks
* Add user authentication and multi-driver profiles

---

## 👨‍💻 Author

**Vihsal Singh**
AI/ML Developer | Computer Vision Researcher

* 💼 IBM x Casbox SkillsBuild Intern (AIML)
* 🎓 Research on Pneumonia Detection using Deep Learning
* 🧑‍💻 Exploring AI-based automation for real-world applications

---
