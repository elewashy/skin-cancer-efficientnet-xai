import io
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
from torchvision import models, transforms

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "training_artifacts" / "efficientnet_b0_gpu_best.pt"
DATA_DIR   = BASE_DIR / "data"

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Load model once at startup ─────────────────────────────────────────────────
print("Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

CLASSES     = checkpoint["classes"]          # ['benign', 'malignant']
IMAGE_SIZE  = checkpoint["image_size"]       # 224
MEAN        = checkpoint["mean"]
STD         = checkpoint["std"]

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"Model loaded — classes: {CLASSES}")

# Eval transform — identical to training eval_transform
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def predict_image(pil_image: Image.Image) -> dict:
    """Run inference on a PIL image. Returns label, confidence, and all probs."""
    tensor = eval_transform(pil_image.convert("RGB")).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].numpy()

    pred_idx    = int(np.argmax(probs))
    pred_label  = CLASSES[pred_idx]
    confidence  = float(probs[pred_idx])

    return {
        "label":      pred_label,
        "confidence": round(confidence * 100, 2),
        "benign_prob":    round(float(probs[0]) * 100, 2),
        "malignant_prob": round(float(probs[1]) * 100, 2),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    try:
        pil_image = Image.open(file.stream)
        result    = predict_image(pil_image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-path", methods=["POST"])
def predict_path():
    """Predict from a server-side file path (used for sample images)."""
    data = request.get_json()
    if not data or "path" not in data:
        return jsonify({"error": "No path provided"}), 400
    image_path = Path(data["path"])
    if not image_path.exists():
        return jsonify({"error": "File not found"}), 404
    try:
        pil_image = Image.open(image_path)
        result    = predict_image(pil_image)
        result["true_label"] = image_path.parent.name  # benign or malignant
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/samples")
def samples():
    """Return a list of sample images from data/test/, 8 per class."""
    result = {"benign": [], "malignant": []}
    for cls in ["benign", "malignant"]:
        cls_dir = DATA_DIR / "test" / cls
        if not cls_dir.exists():
            continue
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png"))
        picked = random.sample(images, min(8, len(images)))
        result[cls] = [str(p) for p in picked]
    return jsonify(result)


@app.route("/image")
def serve_image():
    """Serve a local image file by absolute path (for previewing sample images)."""
    path = request.args.get("path", "")
    p    = Path(path)
    if not p.exists() or not p.is_file():
        return "Not found", 404
    # Safety: only serve from inside the project data directory
    try:
        p.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        return "Forbidden", 403
    return send_file(p, mimetype="image/jpeg")


@app.route("/metrics")
def metrics():
    """Return model test metrics from the CSV."""
    import csv
    metrics_path = BASE_DIR / "training_artifacts" / "model_comparison_metrics.csv"
    rows = []
    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return jsonify(rows)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
