import os
import json
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import time

# ===================================================================
#  APP INIT
# ===================================================================

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

TFLITE_DIR = "models"
os.makedirs(TFLITE_DIR, exist_ok=True)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Food Classification dataset")
ALLOWED_IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# Google Cloud for custom + VGG models
GCS_BUCKET = "https://storage.googleapis.com/food-classification-models-bucket"

# ResNet backend API
RESNET_API_URL = "https://web-production-ba634.up.railway.app/predict"

# ===================================================================
#  NORMALIZER
# ===================================================================

def normalize(x: str):
    return x.lower().strip().replace(" ", "_")

# ===================================================================
#  MODEL → CLASS INDEX (Custom + VGG)
# ===================================================================

RAW_MODEL_CLASS_INDEX = {
    "custom_model_1":  {'apple_pie': 0, 'baked_potato': 1, 'burger': 2},
    "custom_model_2":  {'butter_naan': 0, 'chai': 1, 'chapati': 2},
    "custom_model_3":  {'cheesecake': 0, 'chicken_curry': 1, 'chole_bhature': 2},
    "custom_model_4":  {'crispy_chicken': 0, 'dal_makhani': 1, 'dhokla': 2},
    "custom_model_5":  {'donut': 0, 'fried_rice': 1, 'fries': 2},
    "custom_model_6":  {'hot_dog': 0, 'ice_cream': 1, 'idli': 2},
    "custom_model_7":  {'jalebi': 0, 'kaathi_rolls': 1, 'kadai_paneer': 2},
    "custom_model_8":  {'kulfi': 0, 'masala_dosa': 1, 'momos': 2},
    "custom_model_9":  {'omelette': 0, 'paani_puri': 1, 'pakode': 2},
    "custom_model_10": {'pav_bhaji': 0, 'pizza': 1, 'samosa': 2},
    "custom_model_11": {'sandwich': 0, 'sushi': 1, 'taco': 2, 'taquito': 3},

    "vgg_model_1":  {'apple_pie': 0, 'baked_potato': 1, 'burger': 2},
    "vgg_model_2":  {'butter_naan': 0, 'chai': 1, 'chapati': 2},
    "vgg_model_3":  {'cheesecake': 0, 'chicken_curry': 1, 'chole_bhature': 2},
    "vgg_model_4":  {'crispy_chicken': 0, 'dal_makhani': 1, 'dhokla': 2},
    "vgg_model_5":  {'donut': 0, 'fried_rice': 1, 'fries': 2},
    "vgg_model_6":  {'hot_dog': 0, 'ice_cream': 1, 'idli': 2},
    "vgg_model_7":  {'jalebi': 0, 'kaathi_rolls': 1, 'kadai_paneer': 2},
    "vgg_model_8":  {'kulfi': 0, 'masala_dosa': 1, 'momos': 2},
    "vgg_model_9":  {'omelette': 0, 'paani_puri': 1, 'pakode': 2},
    "vgg_model_10": {'pav_bhaji': 0, 'pizza': 1, 'samosa': 2},
    "vgg_model_11": {'sandwich': 0, 'sushi': 1, 'taco': 2, 'taquito': 3},

    "resnet_model_1":  {'apple_pie': 0, 'baked_potato': 1, 'burger': 2},
    "resnet_model_2":  {'butter_naan': 0, 'chai': 1, 'chapati': 2},
    "resnet_model_3":  {'cheesecake': 0, 'chicken_curry': 1, 'chole_bhature': 2},
    "resnet_model_4":  {'crispy_chicken': 0, 'dal_makhani': 1, 'dhokla': 2},
    "resnet_model_5":  {'donut': 0, 'fried_rice': 1, 'fries': 2},
    "resnet_model_6":  {'hot_dog': 0, 'ice_cream': 1, 'idli': 2},
    "resnet_model_7":  {'jalebi': 0, 'kaathi_rolls': 1, 'kadai_paneer': 2},
    "resnet_model_8":  {'kulfi': 0, 'masala_dosa': 1, 'momos': 2},
    "resnet_model_9":  {'omelette': 0, 'paani_puri': 1, 'pakode': 2},
    "resnet_model_10": {'pav_bhaji': 0, 'pizza': 1, 'samosa': 2},
    "resnet_model_11": {'sandwich': 0, 'sushi': 1, 'taco': 2, 'taquito': 3},
}

MODEL_CLASS_INDEX = {
    m.lower(): {normalize(k): v for k, v in mapping.items()}
    for m, mapping in RAW_MODEL_CLASS_INDEX.items()
}

# ===================================================================
#  CLASS JSON
# ===================================================================

CLASS_JSON = json.load(open("class.json"))
CLASS_LIST = list(CLASS_JSON.keys())

# ===================================================================
#  MODEL METRICS — Custom + VGG + ResNet
# ===================================================================

MODEL_EVAL_JSON = {
    "custom_model": json.load(open("model_evaluation_results.json")),
    "vgg_model": json.load(open("model_evaluation_results_vgg.json")),
    "resnet_model": json.load(open("model_evaluation_results_resnet.json")),
}

for mtype, data in MODEL_EVAL_JSON.items():
    MODEL_EVAL_JSON[mtype] = {normalize(k): v for k, v in data.items()}

# ===================================================================
#  TFLITE LOADING — only one model at a time
# ===================================================================

tflite_cache = {}

def clear_tflite_folder():
    for f in os.listdir(TFLITE_DIR):
        if f.endswith(".tflite"):
            try:
                os.remove(os.path.join(TFLITE_DIR, f))
            except:
                pass

def download_tflite_from_gcs(model_name):
    clear_tflite_folder()
    url = f"{GCS_BUCKET}/{model_name}.tflite"
    save_path = os.path.join(TFLITE_DIR, f"{model_name}.tflite")

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return save_path
    except:
        return None

def load_tflite_model(model_name):
    global tflite_cache
    tflite_cache = {}

    model_path = os.path.join(TFLITE_DIR, model_name + ".tflite")
    if not os.path.exists(model_path):
        ok = download_tflite_from_gcs(model_name)
        if ok is None:
            return None

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    tflite_cache[model_name] = interpreter
    return interpreter

# ===================================================================
#  IMAGE PREPROCESS
# ===================================================================

def preprocess(img, size):
    img = img.resize((size, size))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

# ===================================================================
#  ROUTES
# ===================================================================

@app.route("/")
def index():
    return render_template("index.html", classes=CLASS_LIST)


@app.route("/class_sample/<path:class_name>")
def class_sample(class_name):
    """Serve one sample image for a food class from the dataset."""
    if not class_name or not os.path.isdir(DATASET_DIR):
        return "", 404
    cnorm = normalize(class_name)
    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        if normalize(folder) != cnorm:
            continue
        for f in os.listdir(folder_path):
            if f.endswith(ALLOWED_IMG_EXT):
                return send_file(
                    os.path.join(folder_path, f),
                    mimetype="image/jpeg",
                    as_attachment=False,
                )
        break
    return "", 404


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    img_file = request.files["file"]
    fname = secure_filename(img_file.filename)
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    img_file.save(fpath)

    try:
        model_type = request.form.get("model_type", "").lower()
        selected_class = request.form.get("selected_class", "")
        cname = normalize(selected_class)

        # ======================================================
        # ⭐ RESNET: try local TFLite first, then external API
        # ======================================================
        if model_type == "resnet_model":
            class_info = MODEL_EVAL_JSON["resnet_model"].get(cname)
            if class_info is not None:
                model_used = class_info["model_used"].lower()
                interpreter = load_tflite_model(model_used)
                if interpreter is not None:
                    input_info = interpreter.get_input_details()[0]
                    _, h, w, _ = input_info["shape"]
                    img = Image.open(fpath).convert("RGB")
                    x = preprocess(img, w)
                    interpreter.set_tensor(input_info["index"], x)
                    interpreter.invoke()
                    output_info = interpreter.get_output_details()[0]
                    preds = interpreter.get_tensor(output_info["index"]).squeeze().astype(float)
                    idx = int(np.argmax(preds))
                    confidence = float(np.max(preds))
                    class_map = MODEL_CLASS_INDEX[model_used]
                    predicted_label = next(k for k, v in class_map.items() if v == idx)
                    predicted_norm = normalize(predicted_label)
                    predicted_info = MODEL_EVAL_JSON["resnet_model"].get(predicted_norm, {})
                    labels = list(class_map.keys())
                    matrix = predicted_info.get("confusion_matrix_full", [])
                    return jsonify({
                        "success": True,
                        "selected_class": selected_class,
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                        "accuracy": predicted_info.get("accuracy", "NA"),
                        "precision": predicted_info.get("precision", "NA"),
                        "recall": predicted_info.get("recall", "NA"),
                        "f1_score": predicted_info.get("f1_score", "NA"),
                        "confusion_matrix_labels": labels,
                        "confusion_matrix_full": matrix,
                        "model_used": model_used,
                    })

            # Fallback: external ResNet API (re-read file each attempt for retries)
            file_bytes = None
            for attempt in range(3):
                print(f"[ResNet API] Attempt {attempt+1}/3")
                try:
                    with open(fpath, "rb") as f:
                        file_bytes = f.read()
                    resp = requests.post(
                        RESNET_API_URL,
                        files={"file": (fname, file_bytes, img_file.mimetype or "image/jpeg")},
                        data={"selected_class": selected_class},
                        timeout=300,
                    )
                except Exception as e:
                    print("ResNet API error:", e)
                    time.sleep(5 if attempt < 2 else 0)
                    continue

                if resp.status_code in (502, 503):
                    time.sleep(10)
                    continue

                if resp.status_code == 200:
                    res = resp.json()
                    metrics = res.get("metrics", {})
                    resnet_model_used = res.get("model_used", "resnet_model")
                    model_num = str(resnet_model_used).split("_")[-1]
                    vgg_model_name = f"vgg_model_{model_num}".lower()
                    confusion_labels = list(MODEL_CLASS_INDEX.get(vgg_model_name, {}).keys()) or list(metrics.get("confusion_matrix_full", [[]])[0])
                    return jsonify({
                        "success": True,
                        "selected_class": selected_class,
                        "predicted_label": res.get("predicted_label"),
                        "confidence": res.get("confidence"),
                        "model_used": resnet_model_used,
                        "accuracy": metrics.get("accuracy", "NA"),
                        "precision": metrics.get("precision", "NA"),
                        "recall": metrics.get("recall", "NA"),
                        "f1_score": metrics.get("f1_score", "NA"),
                        "true_positive": metrics.get("true_positive", "NA"),
                        "false_positive": metrics.get("false_positive", "NA"),
                        "false_negative": metrics.get("false_negative", "NA"),
                        "true_negative": metrics.get("true_negative", "NA"),
                        "confusion_matrix_labels": confusion_labels,
                        "confusion_matrix_full": metrics.get("confusion_matrix_full", []),
                    })

                time.sleep(5)

            return jsonify({"success": False, "error": "ResNet failed (local TFLite not found and API unavailable). Add resnet_model_*.tflite to models/ or GCS."})

        # ======================================================
        # ⭐ LOCAL — CUSTOM / VGG MODELS
        # ======================================================
        if model_type not in MODEL_EVAL_JSON:
            return jsonify({"success": False, "error": f"Invalid model_type {model_type}"}), 400

        class_info = MODEL_EVAL_JSON[model_type].get(cname)
        if class_info is None:
            return jsonify({"success": False, "error": f"Class not found: {cname}"}), 400

        model_used = class_info["model_used"].lower()

        interpreter = load_tflite_model(model_used)
        if interpreter is None:
            return jsonify({"success": False, "error": "TFLite load failed"}), 500

        input_info = interpreter.get_input_details()[0]
        _, h, w, _ = input_info["shape"]

        img = Image.open(fpath).convert("RGB")
        x = preprocess(img, w)

        interpreter.set_tensor(input_info["index"], x)
        interpreter.invoke()

        output = interpreter.get_output_details()[0]
        preds = interpreter.get_tensor(output["index"]).squeeze().astype(float)

        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        class_map = MODEL_CLASS_INDEX[model_used]
        predicted_label = next(k for k, v in class_map.items() if v == idx)

        predicted_norm = normalize(predicted_label)
        predicted_info = MODEL_EVAL_JSON[model_type].get(predicted_norm, {})

        labels = list(class_map.keys())
        matrix = predicted_info.get("confusion_matrix_full", [])

        return jsonify({
            "success": True,
            "selected_class": selected_class,
            "predicted_label": predicted_label,
            "confidence": confidence,

            "accuracy": predicted_info.get("accuracy", "NA"),
            "precision": predicted_info.get("precision", "NA"),
            "recall": predicted_info.get("recall", "NA"),
            "f1_score": predicted_info.get("f1_score", "NA"),

            "confusion_matrix_labels": labels,
            "confusion_matrix_full": matrix,

            "model_used": model_used
        })

    finally:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except:
                pass

# ===================================================================
#  MAIN
# ===================================================================

if __name__ == "__main__":
    app.run(debug=True)
