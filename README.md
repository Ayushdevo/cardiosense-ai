# ❤️ CardioSense AI

### Deep Learning-Based ECG Analysis System

CardioSense AI is a web-based application that analyzes ECG (Electrocardiogram) signals using a Convolutional Neural Network (CNN). It detects abnormalities, calculates heart rate, visualizes ECG signals, and converts raw ECG data into CSV format.

---

## 🔥 Features

* 🧠 AI-based ECG Classification (Normal / Abnormal)
* ❤️ Heart Rate (BPM) Calculation using R-Peak Detection
* 📈 ECG Waveform Visualization with Peak Marking
* 🔄 Raw ECG Data → CSV Converter
* 🌐 Interactive Web Interface (Flask)

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras (Deep Learning)
* Flask (Backend)
* Pandas, NumPy (Data Processing)
* SciPy (Signal Processing)
* Matplotlib (Visualization)
* HTML, CSS, JavaScript (Frontend)
* Render (Deployment)
* GitHub (Version Control)

---

## 📂 Project Structure

```
cardiosense-ai/
│
├── app.py
├── ecg_cnn_model.h5
├── requirements.txt
├── runtime.txt
├── Procfile
│
└── templates/
    └── index.html
```

---

## ⚙️ Installation (Run Locally)

```bash
git clone https://github.com/Ayushdevo/cardiosense-ai.git
cd cardiosense-ai
pip install -r requirements.txt
python app.py
```

Open in browser:
http://127.0.0.1:5000/

---

## 🚀 Deployment (Render)

1. Push code to GitHub
2. Go to Render → New Web Service
3. Connect repository
4. Set:

Build Command:

```
pip install -r requirements.txt
```

Start Command:

```
gunicorn app:app --workers=1 --threads=1 --timeout=120
```

5. Deploy

---

## 🧠 How It Works

1. Upload ECG CSV file
2. Preprocessing:

   * Normalize signal
   * Extract segment
3. CNN Model:

   * Predict Normal / Abnormal
4. Signal Processing:

   * Detect peaks (SciPy)
   * Compute heart rate
5. Output:

   * Prediction + Confidence
   * ECG waveform graph

---

## 📊 Model Details

* Model: Convolutional Neural Network (CNN)
* Input: ECG signal segment
* Output: Binary classification
* Dataset: MIT-BIH Arrhythmia Dataset

---

## ⚠️ Limitations

* Free hosting may be slow (cold start)
* High memory usage due to TensorFlow
* Not intended for medical diagnosis

---

## 👨‍💻 Author

**Ayush Tiwari**
ECE + AI | IIT Guwahati

---

## 📌 Future Improvements

* Real-time ECG monitoring
* Mobile app integration
* Lightweight model optimization
* Multi-class arrhythmia detection

---

## ⭐ Acknowledgements

* MIT-BIH ECG Database
* TensorFlow & Keras
* SciPy Signal Processing

---

## 🚀 Live Demo

👉 https://cardiosense-ai-67e1.onrender.com
