import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
import os
import traceback

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
    data = pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip')
    data.columns = data.columns.astype(str).str.replace("'", "").str.strip().str.upper()

    if data.shape[1] < 1:
        return None, "The uploaded CSV has no columns."

    if 'MLII' in data.columns:
        signal = data['MLII'].values
    elif data.shape[1] >= 2:
        signal = data.iloc[:, 1].values
    else:
        signal = data.iloc[:, 0].values

    # Force the signal to be numeric (turns text/headers mixed in data into NaN)
    signal = pd.to_numeric(signal, errors='coerce')
    
    # Drop any NaN values caused by text
    signal = signal[~np.isnan(signal)]

    if len(signal) < 400:
        return None, f"Not enough valid numerical data points. Found {len(signal)}, need at least 400."

    # Normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    segment = signal[:200].reshape(1, 200, 1)

    return (segment, signal), None


# ----------- PREDICT -----------
def predict_ecg(file):
    try:
        # Check if upload timed out or is empty
        if file is None:
            return "❌ No file uploaded. Please ensure the file finishes uploading before clicking Analyze.", None

        # Safely extract the file path depending on the Gradio version
        if isinstance(file, str):
            file_path = file
        elif isinstance(file, dict):
            file_path = file.get("path") or file.get("name")
        elif hasattr(file, "path"):
            file_path = file.path
        elif hasattr(file, "name"):
            file_path = file.name
        else:
            return f"❌ Unknown file object type: {type(file)}", None

        # Run preprocessing
        result_data, error_msg = preprocess(file_path)

        if result_data is None:
            return f"❌ Invalid ECG file.\n\n**Reason:** {error_msg}", None

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

        # Save to disk to prevent BytesIO garbage collection errors
        output_image_path = "temp_ecg_plot.png"
        plt.savefig(output_image_path, format='png')
        plt.close()

        final_img = Image.open(output_image_path)

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

        return output_text, final_img

    except Exception as e:
        # If the app crashes, output the exact error to the UI 
        error_trace = traceback.format_exc()
        return f"❌ **Application Error:**\n```\n{str(e)}\n```\n\n**Traceback:**\n```\n{error_trace}\n```", None


# ----------- GRADIO UI -----------
with gr.Blocks(title="CardioSense AI") as demo:
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
            plot_output = gr.Image(label="📈 ECG Graph")

    predict_btn.click(
        fn=predict_ecg,
        inputs=file_input,
        outputs=[result_output, plot_output]
    )

    gr.Markdown("> ⚠️ For educational use only. Not a substitute for medical advice.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Soft()
    )
