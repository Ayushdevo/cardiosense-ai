import gradio as gr
import numpy as np
import pandas as pd
import base64
from io import BytesIO

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

# ----------- LOAD MODEL -----------
model = tf.keras.models.load_model("ecg_cnn_model.h5", compile=False)


# ----------- PREPROCESS -----------
def preprocess(filepath):
    try:
        data = pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip')
    except:
        return None

    data.columns = data.columns.str.replace("'", "").str.strip().str.upper()

    if data.shape[1] < 1:
        return None

    if 'MLII' in data.columns:
        signal = data['MLII'].values
    elif data.shape[1] >= 2:
        signal = data.iloc[:, 1].values
    else:
        signal = data.iloc[:, 0].values

    if len(signal) < 400:
        return None

    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    segment = signal[:200].reshape(1, 200, 1)

    return segment, signal


# ----------- PREDICT -----------
def predict_ecg(file):
    if file is None:
        return "❌ No file uploaded.", None

    result_data = preprocess(file.name)

    if result_data is None:
        return "❌ Invalid ECG file. Make sure it has at least 400 rows.", None

    segment, signal = result_data

    # Model prediction
    pred = model.predict(segment, verbose=0)[0][0]
    confidence = round(float(pred) * 100, 2) if pred > 0.5 else round((1 - float(pred)) * 100, 2)
    result = "Abnormal" if pred > 0.5 else "Normal"

    # Peak detection
    peaks, _ = find_peaks(signal, distance=200, height=0.5)

    # Heart rate
    if len(peaks) > 1 and np.mean(np.diff(peaks)) > 0:
        rr = np.diff(peaks) / 360
        heart_rate = int(60 / np.mean(rr))
    else:
        heart_rate = None

    # ECG Plot
    plt.figure(figsize=(8, 3))
    plt.plot(signal[:500], color='royalblue', label="ECG Signal")
    for p in peaks:
        if p < 500:
            plt.plot(p, signal[p], "ro", markersize=5)
    bpm_label = f"{heart_rate} BPM" if heart_rate else "-- BPM"
    plt.title(f"ECG Signal ({bpm_label})", fontsize=13)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Result text
    emoji = "🔴" if result == "Abnormal" else "🟢"
    advice = "Please consult a cardiologist." if result == "Abnormal" else "Your ECG looks healthy."
    output_text = (
        f"{emoji} Result: **{result}**\n"
        f"📊 Confidence: {confidence}%\n"
        f"❤️ Heart Rate: {heart_rate if heart_rate else 'N/A'} BPM\n\n"
        f"⚠️ {advice}\n\n"
        f"_This tool is for educational purposes only. Always consult a doctor._"
    )

    return output_text, img


# ----------- GRADIO UI -----------
with gr.Blocks(theme=gr.themes.Soft(), title="CardioSense AI") as demo:
    gr.Markdown("""
    # 💓 CardioSense AI
    ### ECG Heart Condition Detector
    Upload your ECG data as a **CSV file** to detect if it is **Normal** or **Abnormal**.
    """)

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="📂 Upload ECG CSV File", file_types=[".csv"])
            predict_btn = gr.Button("🔍 Analyze ECG", variant="primary")

        with gr.Column():
            result_output = gr.Markdown(label="Result")
            plot_output = gr.Image(label="📈 ECG Graph", type="filepath")

    predict_btn.click(
        fn=predict_ecg,
        inputs=file_input,
        outputs=[result_output, plot_output]
    )

    gr.Markdown("> ⚠️ For educational use only. Not a substitute for medical advice.")

demo.launch()
