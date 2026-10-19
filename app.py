
import os
import io
import base64
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, request, jsonify, render_template_string
from scipy.signal import find_peaks
from PIL import Image

from ecg_processing import binary_probability, prepare_signal
from ecg_plot import render_ecg_plot

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    logger.warning("TensorFlow devices were already initialized.")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ecg_cnn_model.h5"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"ECG model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

# Required by Flask/Vercel.
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


# --------------------------------------------------
# PREPROCESS ECG
# --------------------------------------------------

def preprocess(filepath):
    try:
        data = pd.read_csv(filepath, encoding="latin1")

        data.columns = (
            data.columns.astype(str)
            .str.replace("'", "", regex=False)
            .str.strip()
            .str.upper()
        )

        if data.empty or len(data.columns) == 0:
            return None, "The CSV file is empty."

        if "MLII" in data.columns:
            signal = data["MLII"].to_numpy()
        elif data.shape[1] >= 2:
            signal = data.iloc[:, 1].to_numpy()
        else:
            signal = data.iloc[:, 0].to_numpy()

        signal = pd.to_numeric(
            pd.Series(signal), errors="coerce"
        ).to_numpy(dtype=np.float32)

        signal = signal[np.isfinite(signal)]

        if signal.size == 0:
            return None, "No valid numeric ECG samples were found."

        prepared = prepare_signal(
            signal, model.input_shape[1]
        )

        return prepared, None

    except (ValueError, TypeError) as exc:
        logger.warning("Invalid ECG data: %s", exc)
        return None, str(exc)

    except Exception:
        logger.exception("Unable to preprocess ECG")
        return None, "Unable to read or preprocess this CSV file."


# --------------------------------------------------
# IMAGE CONVERSION
# --------------------------------------------------

def image_to_data_uri(result):
    """Convert common plot return types into a browser image."""
    if result is None:
        return None

    if isinstance(result, (str, os.PathLike)):
        path = Path(result)
        if path.is_file():
            raw = path.read_bytes()
            suffix = path.suffix.lower()
            mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
            return f"data:{mime};base64,{base64.b64encode(raw).decode()}"

    buffer = io.BytesIO()

    if isinstance(result, bytes):
        raw = result
    elif hasattr(result, "savefig"):
        result.savefig(buffer, format="png", bbox_inches="tight")
        raw = buffer.getvalue()
    elif hasattr(result, "save"):
        result.save(buffer, format="PNG")
        raw = buffer.getvalue()
    else:
        array = np.asarray(result)

        if array.ndim not in (2, 3):
            raise ValueError("Unsupported ECG plot format.")

        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                if array.size and array.min() >= 0 and array.max() <= 1:
                    array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)

        raw_image = Image.fromarray(array)
        raw_image.save(buffer, format="PNG")
        raw = buffer.getvalue()

    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# --------------------------------------------------
# ECG ANALYSIS
# --------------------------------------------------


def analyze_ecg(filepath):
    result_data, error = preprocess(filepath)

    if result_data is None:
        raise ValueError(error or "Invalid ECG file.")

    segment, signal = result_data

    prediction = binary_probability(
        model.predict(segment, verbose=0)
    )
    prediction = float(np.asarray(prediction).reshape(-1)[0])

    if not np.isfinite(prediction):
        raise ValueError("The model returned an invalid prediction.")

    prediction = float(np.clip(prediction, 0.0, 1.0))

    result = "Abnormal" if prediction > 0.5 else "Normal"
    confidence = (
        prediction * 100
        if prediction > 0.5
        else (1 - prediction) * 100
    )

    signal = np.asarray(signal).reshape(-1)
    peaks, _ = find_peaks(signal, distance=200, height=0.5)

    heart_rate = None
    if len(peaks) > 1:
        mean_rr = float(np.mean(np.diff(peaks))) / 360.0
        if mean_rr > 0:
            heart_rate = int(round(60 / mean_rr))

    plot = render_ecg_plot(signal, peaks, heart_rate)

    return {
        "result": result,
        "confidence": round(confidence, 2),
        "heart_rate": heart_rate,
        "plot": image_to_data_uri(plot),
        "advice": (
            "Please consult a qualified healthcare professional."
            if result == "Abnormal"
            else "This result does not rule out a heart condition."
        ),
    }



# --------------------------------------------------
# WEB UI
# --------------------------------------------------

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CardioSense AI</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 28px 16px; min-height: 100vh;
      font-family: system-ui, sans-serif; color: #172033;
      background: linear-gradient(135deg, #eef5ff, #f8faff);
    }
    main {
      max-width: 760px; margin: auto; padding: 28px;
      background: white; border-radius: 20px;
      box-shadow: 0 12px 40px #20305015;
    }
    h1 { margin-top: 0; color: #173e75; }
    .muted { color: #637087; line-height: 1.6; }
    form { margin: 24px 0; }
    input[type=file] {
      width: 100%; padding: 16px; border: 1px dashed #8da8cc;
      border-radius: 12px; background: #f8fbff;
    }
    button {
      margin-top: 14px; padding: 13px 20px; border: 0;
      border-radius: 10px; background: #2459c4; color: white;
      font-weight: 650; cursor: pointer; width: 100%;
    }
    button:disabled { opacity: .6; cursor: wait; }
    .result {
      margin-top: 22px; padding: 18px; border-radius: 12px;
      background: #f4f7fb; overflow-wrap: anywhere;
    }
    img { width: 100%; height: auto; margin-top: 18px; border-radius: 10px; }
    .error { color: #a12626; }
    footer { font-size: 13px; margin-top: 24px; }
  </style>
</head>
<body>
<main>
  <h1>💓 CardioSense AI</h1>
  <p class="muted">
    Upload an ECG CSV file to run an experimental model analysis.
    The file must contain numeric ECG samples.
  </p>

  <form action="/analyze" method="post" enctype="multipart/form-data">
    <input type="file" name="file" accept=".csv,text/csv" required>
    <button type="submit">🔍 Analyze ECG</button>
  </form>

  {% if error %}
    <div class="result error"><strong>Analysis failed:</strong> {{ error }}</div>
  {% endif %}

  {% if result %}
    <section class="result">
      <h2>{{ result.result }}</h2>
      <p><strong>Model score:</strong> {{ result.confidence }}%</p>
      <p><strong>Estimated heart rate:</strong>
        {{ result.heart_rate if result.heart_rate is not none else "N/A" }} BPM
      </p>
      <p>{{ result.advice }}</p>
      {% if result.plot %}
        <img src="{{ result.plot }}" alt="ECG waveform">
      {% endif %}
    </section>
  {% endif %}

  <footer class="muted">
    Educational use only. This tool is not a medical device and cannot
    diagnose or exclude a heart condition. Seek professional medical care
    for symptoms or concerns.
  </footer>
</main>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(PAGE)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.post("/analyze")
def analyze():
    uploaded = request.files.get("file")

    if uploaded is None or not uploaded.filename:
        return render_template_string(
            PAGE, error="Please select a CSV file."
        ), 400

    if Path(uploaded.filename).suffix.lower() != ".csv":
        return render_template_string(
            PAGE, error="Only CSV files are accepted."
        ), 400

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", delete=False
        ) as temp:
            uploaded.save(temp)
            temp_path = temp.name

        result = analyze_ecg(temp_path)
        return render_template_string(PAGE, result=result)

    except ValueError as exc:
        return render_template_string(PAGE, error=str(exc)), 400

    except Exception:
        logger.exception("ECG analysis request failed")
        return render_template_string(
            PAGE,
            error="Analysis failed. Check the CSV format and server logs."
        ), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/analyze")
def analyze_api():
    """JSON API endpoint for clients that submit multipart form data."""
    uploaded = request.files.get("file")

    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "No CSV file uploaded."}), 400

    if Path(uploaded.filename).suffix.lower() != ".csv":
        return jsonify({"error": "Only CSV files are accepted."}), 400

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", delete=False
        ) as temp:
            uploaded.save(temp)
            temp_path = temp.name

        result = analyze_ecg(temp_path)
        return jsonify(result)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        logger.exception("ECG API request failed")
        return jsonify({"error": "ECG analysis failed."}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": "File exceeds the 10 MB upload limit."
    }), 413


# Local development only. Vercel imports `app` directly.
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        debug=False
    )
