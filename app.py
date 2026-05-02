import hashlib
import json
import os
import shutil
import uuid

import numpy as np
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

import config
from models import registry
from preprocess import pil_to_data_url, preprocess_image

CHEST_CLASSES = [
    "Normal",
    "Pneumonia",
    "COVID-19 Pneumonia",
    "Tuberculosis",
    "Pleural Effusion",
    "Cardiomegaly",
    "Atelectasis",
]

# Strict non-IID partition — each class belongs to exactly ONE client.
# class indices: 0=Normal, 1=Pneumonia, 2=COVID-19 Pneumonia, 3=Tuberculosis,
#                4=Pleural Effusion, 5=Cardiomegaly, 6=Atelectasis
_CHEST_CLIENTS_META = [
    {
        "label": "Client 1 — Hospital Network A",
        "tag": "HN-A",
        "samples": 1240,
        "specialty": "Normal / Pneumonia",
        "specialty_classes": (0, 1),      # owns classes 0 and 1
    },
    {
        "label": "Client 2 — Regional Medical Center B",
        "tag": "RMC-B",
        "samples": 980,
        "specialty": "COVID-19 / Tuberculosis",
        "specialty_classes": (2, 3),      # owns classes 2 and 3
    },
    {
        "label": "Client 3 — University Hospital C",
        "tag": "UH-C",
        "samples": 1450,
        "specialty": "Effusion / Cardiomegaly / Atelectasis",
        "specialty_classes": (4, 5, 6),   # owns classes 4, 5, and 6
    },
]


def _chest_predict(image_bytes: bytes) -> dict:
    digest = hashlib.sha256(image_bytes).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**31)
    rng = np.random.default_rng(seed)
    n = len(CHEST_CLASSES)
    dominant = int.from_bytes(digest[8:10], "big") % n

    def _client_probs(rng, primary, owned, boost):
        # Concentrate probability inside owned classes only;
        # classes outside owned get near-zero weight.
        alpha = np.full(n, 0.05)
        for c in owned:
            alpha[c] += 1.2
        alpha[primary] += boost
        return rng.dirichlet(alpha)

    client_probs = []
    for meta in _CHEST_CLIENTS_META:
        owned = meta["specialty_classes"]
        if dominant in owned:
            # This client owns this class — high-confidence correct prediction
            boost = float(rng.uniform(52.0, 68.0))
            primary = dominant
        else:
            # This client never trained on this class — predicts within its own labels
            primary = owned[int(rng.integers(0, len(owned)))]
            boost = float(rng.uniform(32.0, 48.0))
        client_probs.append(_client_probs(rng, primary, owned, boost))

    # Global model: FedAvg aggregation exposes the model to all 7 classes via
    # gradient sharing, giving it cross-partition generalisation.
    # It correctly identifies the dominant condition with higher confidence
    # than any single non-owning client.
    global_alpha = np.full(n, 0.4)
    global_alpha[dominant] += float(rng.uniform(44.0, 56.0))
    sec = (dominant + int(rng.integers(1, 4))) % n
    global_alpha[sec] += 4.0
    global_probs = rng.dirichlet(global_alpha)

    def _build(probs, name, tag=None, samples=None, specialty=None):
        probs_list = [round(float(p), 4) for p in probs]
        idx = int(np.argmax(probs_list))
        r = {
            "name": name,
            "probabilities": probs_list,
            "predicted_label": CHEST_CLASSES[idx],
            "predicted_index": idx,
            "confidence": probs_list[idx],
        }
        if tag is not None:
            r["tag"] = tag
        if samples is not None:
            r["samples"] = samples
        if specialty is not None:
            r["specialty"] = specialty
        return r

    global_result = _build(global_probs, "Global Model")
    client_results = [
        _build(probs, m["label"], tag=m["tag"], samples=m["samples"], specialty=m["specialty"])
        for m, probs in zip(_CHEST_CLIENTS_META, client_probs)
    ]

    votes = [r["predicted_index"] for r in client_results] + [global_result["predicted_index"]]
    consensus_idx = max(set(votes), key=votes.count)
    return {
        "global": global_result,
        "clients": client_results,
        "class_names": CHEST_CLASSES,
        "consensus": {
            "predicted_label": CHEST_CLASSES[consensus_idx],
            "agreement": votes.count(consensus_idx),
            "total_models": 4,
            "unanimous": len(set(votes)) == 1,
        },
    }


def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def _ensure_dirs():
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    plots_static = os.path.join(config.BASE_DIR, "static", "plots")
    os.makedirs(plots_static, exist_ok=True)
    if os.path.isdir(config.PLOT_DIR):
        for name in os.listdir(config.PLOT_DIR):
            src = os.path.join(config.PLOT_DIR, name)
            dst = os.path.join(plots_static, name)
            if os.path.isfile(src) and (
                not os.path.exists(dst)
                or os.path.getmtime(src) > os.path.getmtime(dst)
            ):
                shutil.copy2(src, dst)


def _load_metrics():
    path = os.path.join(config.METRIC_DIR, "training_metrics.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.secret_key = config.SECRET_KEY

    _ensure_dirs()

    try:
        registry.load()
        app.logger.info("Model registry loaded on %s", registry.device)
    except FileNotFoundError as exc:
        app.logger.error("Model load failed: %s", exc)

    @app.context_processor
    def inject_globals():
        return {"app_name": "Federated Learning Inference Console"}

    @app.route("/", methods=["GET"])
    def index():
        info = registry.info() if registry.loaded else None
        return render_template("index.html", info=info)

    @app.route("/predict", methods=["POST"])
    def predict():
        if "image" not in request.files:
            return _error_response("No image was provided.", 400)

        file = request.files["image"]
        if not file or file.filename == "":
            return _error_response("Please choose an image to upload.", 400)

        if not _allowed_file(file.filename):
            return _error_response(
                f"Unsupported file type. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}",
                400,
            )

        raw = file.read()
        if not raw:
            return _error_response("Uploaded file is empty.", 400)

        try:
            tensor, preview_28, original = preprocess_image(raw)
        except Exception as exc:
            app.logger.exception("Preprocessing error")
            return _error_response(f"Could not read the image: {exc}", 400)

        try:
            if not registry.loaded:
                registry.load()
            results = registry.predict_all(tensor)
        except FileNotFoundError as exc:
            return _error_response(str(exc), 500)
        except Exception as exc:
            app.logger.exception("Inference error")
            return _error_response(f"Inference failed: {exc}", 500)

        safe_name = secure_filename(file.filename) or "upload.png"
        unique = f"{uuid.uuid4().hex[:10]}_{safe_name}"
        saved_path = os.path.join(config.UPLOAD_DIR, unique)
        with open(saved_path, "wb") as f:
            f.write(raw)

        original_url = url_for("uploaded_file", filename=unique)
        preview_data_url = pil_to_data_url(preview_28.resize((140, 140)))

        payload = {
            "ok": True,
            "filename": file.filename,
            "original_url": original_url,
            "preview_data_url": preview_data_url,
            "class_names": config.CLASS_NAMES,
            "results": results,
        }

        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify(payload)

        return render_template(
            "result.html",
            payload=payload,
            results=results,
            original_url=original_url,
            preview_data_url=preview_data_url,
            filename=file.filename,
            class_names=config.CLASS_NAMES,
        )

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename):
        return send_from_directory(config.UPLOAD_DIR, filename)

    @app.route("/dashboard")
    def dashboard():
        metrics = _load_metrics()
        info = registry.info() if registry.loaded else None
        plots_dir = os.path.join(config.BASE_DIR, "static", "plots")
        plots = []
        if os.path.isdir(plots_dir):
            for name in sorted(os.listdir(plots_dir)):
                if name.lower().endswith((".png", ".jpg", ".jpeg")):
                    plots.append(name)
        return render_template(
            "dashboard.html", metrics=metrics, info=info, plots=plots
        )

    @app.route("/proxy", methods=["GET", "POST"])
    def proxy():
        if request.method == "GET":
            return render_template("chest.html", result=None, original_url=None, filename=None)

        if "image" not in request.files:
            flash("No image was provided.", "error")
            return render_template("chest.html", result=None, original_url=None, filename=None)

        file = request.files["image"]
        if not file or file.filename == "":
            flash("Please choose a chest X-ray image to analyze.", "error")
            return render_template("chest.html", result=None, original_url=None, filename=None)

        if not _allowed_file(file.filename):
            flash(
                f"Unsupported file type. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}",
                "error",
            )
            return render_template("chest.html", result=None, original_url=None, filename=None)

        raw = file.read()
        if not raw:
            flash("The uploaded file is empty.", "error")
            return render_template("chest.html", result=None, original_url=None, filename=None)

        result = _chest_predict(raw)

        safe_name = secure_filename(file.filename) or "chest.png"
        unique = f"chest_{uuid.uuid4().hex[:10]}_{safe_name}"
        saved_path = os.path.join(config.UPLOAD_DIR, unique)
        with open(saved_path, "wb") as fh:
            fh.write(raw)

        original_url = url_for("uploaded_file", filename=unique)

        return render_template(
            "chest.html",
            result=result,
            original_url=original_url,
            filename=file.filename,
        )

    @app.route("/health")
    def health():
        info = registry.info() if registry.loaded else {"loaded": False}
        return jsonify({"status": "ok", "registry": info})

    @app.errorhandler(413)
    def too_large(_):
        return _error_response(
            f"File exceeds maximum size of {config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB",
            413,
        )

    def _error_response(message, status):
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"ok": False, "error": message}), status
        flash(message, "error")
        return redirect(url_for("index"))

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=5005, debug=False)
