# ❤️ CardioSense AI

### Deep Learning-Based ECG Analysis System

CardioSense AI analyzes uploaded ECG CSV signals using a Convolutional Neural Network (CNN), estimates heart rate, and plots the waveform in a Gradio interface.

---

## 🔥 Features

* 🧠 AI-based ECG Classification (Normal / Abnormal)
* ❤️ Heart Rate (BPM) Calculation using R-Peak Detection
* 📈 ECG Waveform Visualization with Peak Marking
* 🌐 Interactive Web Interface (Gradio)

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras (Deep Learning)
* Gradio (Web Interface)
* Pandas, NumPy (Data Processing)
* SciPy (Signal Processing)
* Matplotlib (Visualization)
* Render (Deployment)
* GitHub (Version Control)

---

## 📂 Project Structure
~~~
CardioSense-AI/
│
├── app.py
├── ecg_cnn_model.h5
├── requirements.txt
├── render.yaml
└── README.md
~~~
---

## ⚙️ Installation (Run Locally)

```bash
git clone https://github.com/Ayushdevo/cardiosense-ai.git
cd cardiosense-ai
pip install -r requirements.txt
python app.py
```

Open in browser:
http://127.0.0.1:10000/

`PORT` overrides the default port of 10000. The model is loaded relative to
`app.py`, so startup also works when the process has a different working directory.

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
python app.py
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
👉 https://cardiosense-ai-i.onrender.com
