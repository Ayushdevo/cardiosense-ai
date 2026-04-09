from flask import Flask, request, render_template, send_file
import numpy as np
import tensorflow as tf
import pandas as pd
import io

import matplotlib.pyplot as plt
import base64
from io import BytesIO
from scipy.signal import find_peaks

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("ecg_cnn_model.h5")


# ---------------- PREPROCESS ----------------
def preprocess(file):
    try:
        data = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')
    except:
        return None

    # Clean column names
    data.columns = data.columns.str.replace("'", "").str.strip().str.upper()

    if data.shape[1] < 1:
        return None

    # Select ECG signal column
    if 'MLII' in data.columns:
        signal = data['MLII'].values
    elif data.shape[1] >= 2:
        signal = data.iloc[:, 1].values
    else:
        signal = data.iloc[:, 0].values

    # Ensure sufficient length
    if len(signal) < 400:
        return None

    # Normalize safely
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    # Prepare input for model
    segment = signal[:400].reshape(1, 400, 1)

    return segment, signal


# ---------------- MAIN ROUTE ----------------
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            return render_template(
                'index.html',
                result="No file selected",
                confidence=None,
                heart_rate=None
            )

        result_data = preprocess(file)

        if result_data is None:
            return render_template(
                'index.html',
                result="Invalid ECG file",
                confidence=None,
                heart_rate=None
            )

        segment, signal = result_data

        # -------- MODEL PREDICTION --------
        pred = model.predict(segment, verbose=0)[0][0]

        if pred > 0.5:
            confidence = round(float(pred) * 100, 2)
        else:
            confidence = round((1 - float(pred)) * 100, 2)

        result = "Abnormal" if pred > 0.5 else "Normal"

        # -------- PEAK DETECTION --------
        peaks, _ = find_peaks(signal, distance=200, height=0.5)

        # -------- HEART RATE --------
        if len(peaks) > 1 and np.mean(np.diff(peaks)) > 0:
            rr = np.diff(peaks) / 360
            heart_rate = int(60 / np.mean(rr))
        else:
            heart_rate = None

        # -------- ECG PLOT --------
        plt.figure(figsize=(6, 2))
        plt.plot(signal[:1000], label="ECG")

        # Mark peaks
        for p in peaks:
            if p < 1000:
                plt.plot(p, signal[p], "ro")

        plt.title(f"ECG Waveform ({heart_rate if heart_rate else '--'} BPM)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template(
            'index.html',
            result=result,
            confidence=confidence,
            heart_rate=heart_rate,
            plot_url=plot_url
        )

    return render_template(
        'index.html',
        result=None,
        confidence=None,
        heart_rate=None
    )


# ---------------- RAW → CSV CONVERTER ----------------
@app.route('/convert', methods=['POST'])
def convert():
    file = request.files.get('rawfile')

    if not file or file.filename == '':
        return "No file uploaded"

    try:
        content = file.read().decode('latin1').splitlines()

        data = []
        for line in content:
            parts = line.strip().split()
            nums = []
            for x in parts:
                try:
                    nums.append(float(x))
                except:
                    continue
            if nums:
                data.append(nums)

        if len(data) == 0:
            return "Invalid raw data"

        df = pd.DataFrame(data)

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='converted_ecg.csv'
        )

    except:
        return "Conversion failed"


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)