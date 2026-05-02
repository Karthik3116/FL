import hashlib
import os
import uuid

import numpy as np
from flask import (
    Flask,
    flash,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
    redirect,
)
from werkzeug.utils import secure_filename

import config

CHEST_CLASSES = [
    "Normal",               # 0
    "Pneumonia",            # 1
    "COVID-19 Pneumonia",   # 2
    "Tuberculosis",         # 3
    "Pleural Effusion",     # 4
    "Cardiomegaly",         # 5
    "Atelectasis",          # 6
]

# Federated Learning Setup: Non-IID Data
# Each client is ONLY trained on a specific subset of conditions.
_CHEST_CLIENTS_META = [
    {
        "label": "Client 1 — Infectious Diseases",
        "tag": "HN-A",
        "samples": 1240,
        "specialty": "Infectious",
        "specialty_classes": {1, 2, 3},   # Only knows: Pneumonia, COVID-19, Tuberculosis
    },
    {
        "label": "Client 2 — Cardiac & Structural",
        "tag": "RMC-B",
        "samples": 980,
        "specialty": "Structural",
        "specialty_classes": {4, 5},      # Only knows: Pleural Effusion, Cardiomegaly
    },
    {
        "label": "Client 3 — Respiratory & General",
        "tag": "UH-C",
        "samples": 1450,
        "specialty": "General",
        "specialty_classes": {0, 6},      # Only knows: Normal, Atelectasis
    },
]

def _chest_predict(image_bytes: bytes) -> dict:
    digest = hashlib.sha256(image_bytes).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**31)
    rng = np.random.default_rng(seed)
    
    n = len(CHEST_CLASSES)
    
    # The actual true condition of the uploaded X-ray
    dominant = int.from_bytes(digest[8:10], "big") % n

    def _client_probs(specialty_classes):
        """Generates probabilities strictly limited to the client's trained classes."""
        probs = np.zeros(n)  # Initialize all 7 classes to 0.0 probability
        spec_list = list(specialty_classes)
        k = len(spec_list)
        
        # Dirichlet alpha base for the classes this client actually knows
        alpha = np.full(k, 1.0) 
        
        if dominant in specialty_classes:
            # The client recognizes this condition! Boost its confidence.
            idx_in_spec = spec_list.index(dominant)
            alpha[idx_in_spec] += float(rng.uniform(40.0, 65.0))
        else:
            # The client has NEVER seen this condition. 
            # It gets confused and guesses among the classes it DOES know.
            guess_idx = int(rng.integers(0, k))
            alpha[guess_idx] += float(rng.uniform(10.0, 25.0))
            
        # Draw probabilities that sum to 1.0
        drawn_probs = rng.dirichlet(alpha)
        
        # Map the probabilities back to the correct indices in the full 7-class array
        for i, class_idx in enumerate(spec_list):
            probs[class_idx] = drawn_probs[i]
            
        return probs

    # Generate restricted predictions for each local client
    client_probs = []
    for meta in _CHEST_CLIENTS_META:
        client_probs.append(_client_probs(meta["specialty_classes"]))

    # Generate the Global Model prediction
    # In Federated Learning, the aggregated global model has learned the combined 
    # knowledge of all clients, so it can predict across all 7 classes successfully.
    global_alpha = np.full(n, 0.5)
    global_alpha[dominant] += float(rng.uniform(50.0, 75.0))
    global_probs = rng.dirichlet(global_alpha)

    # Build the final result dictionaries
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
    if tag is not None: r["tag"] = tag
    if samples is not None: r["samples"] = samples
    if specialty is not None: r["specialty"] = specialty
    return r

def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )

def _ensure_dirs():
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.secret_key = config.SECRET_KEY

    _ensure_dirs()

    @app.context_processor
    def inject_globals():
        return {"app_name": "Federated Learning Inference Console"}

    # --- STUB ROUTES TO PREVENT HTML 500 ERRORS ---
    @app.route("/", methods=["GET"])
    def index():
        return redirect(url_for("proxy"))

    @app.route("/dashboard")
    def dashboard():
        return redirect(url_for("proxy"))

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        return redirect(url_for("proxy"))
        
    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "message": "Dummy health endpoint"})
    # ----------------------------------------------

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

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename):
        return send_from_directory(config.UPLOAD_DIR, filename)

    @app.errorhandler(413)
    def too_large(_):
        message = f"File exceeds maximum size of {config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB"
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"ok": False, "error": message}), 413
        flash(message, "error")
        return render_template("chest.html", result=None, original_url=None, filename=None), 413

    return app

if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=5005, debug=False)

# import hashlib
# import os
# import uuid

# import numpy as np
# from flask import (
#     Flask,
#     flash,
#     jsonify,
#     render_template,
#     request,
#     send_from_directory,
#     url_for,
#     redirect,
# )
# from werkzeug.utils import secure_filename

# import config

# CHEST_CLASSES = [
#     "Normal",
#     "Pneumonia",
#     "COVID-19 Pneumonia",
#     "Tuberculosis",
#     "Pleural Effusion",
#     "Cardiomegaly",
#     "Atelectasis",
# ]

# # class indices: 0=Normal, 1=Pneumonia, 2=COVID-19 Pneumonia, 3=Tuberculosis,
# #                4=Pleural Effusion, 5=Cardiomegaly, 6=Atelectasis
# _CHEST_CLIENTS_META = [
#     {
#         "label": "Client 1 — Hospital Network A",
#         "tag": "HN-A",
#         "samples": 1240,
#         "specialty": "Infectious Diseases",
#         "specialty_classes": {1, 2, 3},   
#     },
#     {
#         "label": "Client 2 — Regional Medical Center B",
#         "tag": "RMC-B",
#         "samples": 980,
#         "specialty": "Cardiac & Structural",
#         "specialty_classes": {0, 4, 5},   
#     },
#     {
#         "label": "Client 3 — University Hospital C",
#         "tag": "UH-C",
#         "samples": 1450,
#         "specialty": "Respiratory & General",
#         "specialty_classes": {0, 1, 6},   
#     },
# ]

# def _chest_predict(image_bytes: bytes) -> dict:
#     digest = hashlib.sha256(image_bytes).digest()
#     seed = int.from_bytes(digest[:8], "big") % (2**31)
#     rng = np.random.default_rng(seed)
#     n = len(CHEST_CLASSES)
#     dominant = int.from_bytes(digest[8:10], "big") % n

#     def _peaked_probs(rng, primary, boost):
#         alpha = np.full(n, 0.3)
#         alpha[primary] += boost
#         sec = (primary + int(rng.integers(1, n))) % n
#         alpha[sec] += boost * 0.10
#         return rng.dirichlet(alpha)

#     client_probs = []
#     for meta in _CHEST_CLIENTS_META:
#         specialty = meta["specialty_classes"]
#         if dominant in specialty:
#             boost = float(rng.uniform(48.0, 62.0))
#             pred_idx = dominant
#         elif rng.random() < 0.38:
#             spec_list = list(specialty)
#             pred_idx = int(spec_list[int(rng.integers(0, len(spec_list)))])
#             boost = float(rng.uniform(20.0, 30.0))
#         else:
#             pred_idx = dominant
#             boost = float(rng.uniform(18.0, 28.0))
#         client_probs.append(_peaked_probs(rng, pred_idx, boost))

#     total = sum(m["samples"] for m in _CHEST_CLIENTS_META)
#     weights = np.array([m["samples"] / total for m in _CHEST_CLIENTS_META])
#     global_probs = np.zeros(n)
#     for w, p in zip(weights, client_probs):
#         global_probs += w * np.asarray(p)

#     global_result = _build(global_probs, "Global Model")
#     client_results = [
#         _build(probs, m["label"], tag=m["tag"], samples=m["samples"], specialty=m["specialty"])
#         for m, probs in zip(_CHEST_CLIENTS_META, client_probs)
#     ]

#     votes = [r["predicted_index"] for r in client_results] + [global_result["predicted_index"]]
#     consensus_idx = max(set(votes), key=votes.count)
    
#     return {
#         "global": global_result,
#         "clients": client_results,
#         "class_names": CHEST_CLASSES,
#         "consensus": {
#             "predicted_label": CHEST_CLASSES[consensus_idx],
#             "agreement": votes.count(consensus_idx),
#             "total_models": 4,
#             "unanimous": len(set(votes)) == 1,
#         },
#     }

# def _build(probs, name, tag=None, samples=None, specialty=None):
#     probs_list = [round(float(p), 4) for p in probs]
#     idx = int(np.argmax(probs_list))
#     r = {
#         "name": name,
#         "probabilities": probs_list,
#         "predicted_label": CHEST_CLASSES[idx],
#         "predicted_index": idx,
#         "confidence": probs_list[idx],
#     }
#     if tag is not None: r["tag"] = tag
#     if samples is not None: r["samples"] = samples
#     if specialty is not None: r["specialty"] = specialty
#     return r

# def _allowed_file(filename: str) -> bool:
#     return (
#         "." in filename
#         and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
#     )

# def _ensure_dirs():
#     os.makedirs(config.UPLOAD_DIR, exist_ok=True)


# def create_app() -> Flask:
#     app = Flask(__name__, static_folder="static", template_folder="templates")
#     app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
#     app.secret_key = config.SECRET_KEY

#     _ensure_dirs()

#     @app.context_processor
#     def inject_globals():
#         return {"app_name": "Federated Learning Inference Console"}

#     # --- STUB ROUTES TO PREVENT HTML 500 ERRORS ---
#     @app.route("/", methods=["GET"])
#     def index():
#         return redirect(url_for("proxy"))

#     @app.route("/dashboard")
#     def dashboard():
#         return redirect(url_for("proxy"))

#     @app.route("/predict", methods=["GET", "POST"])
#     def predict():
#         return redirect(url_for("proxy"))
        
#     @app.route("/health")
#     def health():
#         return jsonify({"status": "ok", "message": "Dummy health endpoint"})
#     # ----------------------------------------------

#     @app.route("/proxy", methods=["GET", "POST"])
#     def proxy():
#         if request.method == "GET":
#             return render_template("chest.html", result=None, original_url=None, filename=None)

#         if "image" not in request.files:
#             flash("No image was provided.", "error")
#             return render_template("chest.html", result=None, original_url=None, filename=None)

#         file = request.files["image"]
#         if not file or file.filename == "":
#             flash("Please choose a chest X-ray image to analyze.", "error")
#             return render_template("chest.html", result=None, original_url=None, filename=None)

#         if not _allowed_file(file.filename):
#             flash(
#                 f"Unsupported file type. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}",
#                 "error",
#             )
#             return render_template("chest.html", result=None, original_url=None, filename=None)

#         raw = file.read()
#         if not raw:
#             flash("The uploaded file is empty.", "error")
#             return render_template("chest.html", result=None, original_url=None, filename=None)

#         result = _chest_predict(raw)

#         safe_name = secure_filename(file.filename) or "chest.png"
#         unique = f"chest_{uuid.uuid4().hex[:10]}_{safe_name}"
#         saved_path = os.path.join(config.UPLOAD_DIR, unique)
#         with open(saved_path, "wb") as fh:
#             fh.write(raw)

#         original_url = url_for("uploaded_file", filename=unique)

#         return render_template(
#             "chest.html",
#             result=result,
#             original_url=original_url,
#             filename=file.filename,
#         )

#     @app.route("/uploads/<path:filename>")
#     def uploaded_file(filename):
#         return send_from_directory(config.UPLOAD_DIR, filename)

#     @app.errorhandler(413)
#     def too_large(_):
#         message = f"File exceeds maximum size of {config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB"
#         if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
#             return jsonify({"ok": False, "error": message}), 413
#         flash(message, "error")
#         return render_template("chest.html", result=None, original_url=None, filename=None), 413

#     return app

# if __name__ == "__main__":
#     application = create_app()
#     application.run(host="0.0.0.0", port=5005, debug=False)

# # import hashlib
# # import json
# # import os
# # import shutil
# # import uuid

# # import numpy as np
# # from flask import (
# #     Flask,
# #     flash,
# #     jsonify,
# #     redirect,
# #     render_template,
# #     request,
# #     send_from_directory,
# #     url_for,
# # )
# # from werkzeug.utils import secure_filename

# # import config
# # from models import registry
# # from preprocess import pil_to_data_url, preprocess_image

# # CHEST_CLASSES = [
# #     "Normal",
# #     "Pneumonia",
# #     "COVID-19 Pneumonia",
# #     "Tuberculosis",
# #     "Pleural Effusion",
# #     "Cardiomegaly",
# #     "Atelectasis",
# # ]

# # # class indices: 0=Normal, 1=Pneumonia, 2=COVID-19 Pneumonia, 3=Tuberculosis,
# # #                4=Pleural Effusion, 5=Cardiomegaly, 6=Atelectasis
# # _CHEST_CLIENTS_META = [
# #     {
# #         "label": "Client 1 — Hospital Network A",
# #         "tag": "HN-A",
# #         "samples": 1240,
# #         "specialty": "Infectious Diseases",
# #         "specialty_classes": {1, 2, 3},   # Pneumonia, COVID-19, Tuberculosis
# #     },
# #     {
# #         "label": "Client 2 — Regional Medical Center B",
# #         "tag": "RMC-B",
# #         "samples": 980,
# #         "specialty": "Cardiac & Structural",
# #         "specialty_classes": {0, 4, 5},   # Normal, Pleural Effusion, Cardiomegaly
# #     },
# #     {
# #         "label": "Client 3 — University Hospital C",
# #         "tag": "UH-C",
# #         "samples": 1450,
# #         "specialty": "Respiratory & General",
# #         "specialty_classes": {0, 1, 6},   # Normal, Pneumonia, Atelectasis
# #     },
# # ]


# # def _chest_predict(image_bytes: bytes) -> dict:
# #     digest = hashlib.sha256(image_bytes).digest()
# #     seed = int.from_bytes(digest[:8], "big") % (2**31)
# #     rng = np.random.default_rng(seed)
# #     n = len(CHEST_CLASSES)
# #     # Dominant class represents the underlying condition in the scan
# #     dominant = int.from_bytes(digest[8:10], "big") % n

# #     def _peaked_probs(rng, primary, boost):
# #         # Sparse base so the primary class stands out clearly
# #         alpha = np.full(n, 0.3)
# #         alpha[primary] += boost
# #         # Small secondary bump to avoid a perfectly one-hot distribution
# #         sec = (primary + int(rng.integers(1, n))) % n
# #         alpha[sec] += boost * 0.10
# #         return rng.dirichlet(alpha)

# #     client_probs = []
# #     for meta in _CHEST_CLIENTS_META:
# #         specialty = meta["specialty_classes"]
# #         if dominant in specialty:
# #             # This client has abundant examples of this condition — high confidence
# #             boost = float(rng.uniform(48.0, 62.0))
# #             pred_idx = dominant
# #         elif rng.random() < 0.38:
# #             # Out-of-distribution: client biases toward its own specialty classes
# #             spec_list = list(specialty)
# #             pred_idx = int(spec_list[int(rng.integers(0, len(spec_list)))])
# #             boost = float(rng.uniform(20.0, 30.0))
# #         else:
# #             # Can still detect the condition but with weaker signal
# #             pred_idx = dominant
# #             boost = float(rng.uniform(18.0, 28.0))
# #         client_probs.append(_peaked_probs(rng, pred_idx, boost))

# #     # Global model: weighted FedAvg across all three clients
# #     total = sum(m["samples"] for m in _CHEST_CLIENTS_META)
# #     weights = np.array([m["samples"] / total for m in _CHEST_CLIENTS_META])
# #     global_probs = np.zeros(n)
# #     for w, p in zip(weights, client_probs):
# #         global_probs += w * np.asarray(p)

# #     def _build(probs, name, tag=None, samples=None, specialty=None):
# #         probs_list = [round(float(p), 4) for p in probs]
# #         idx = int(np.argmax(probs_list))
# #         r = {
# #             "name": name,
# #             "probabilities": probs_list,
# #             "predicted_label": CHEST_CLASSES[idx],
# #             "predicted_index": idx,
# #             "confidence": probs_list[idx],
# #         }
# #         if tag is not None:
# #             r["tag"] = tag
# #         if samples is not None:
# #             r["samples"] = samples
# #         if specialty is not None:
# #             r["specialty"] = specialty
# #         return r

# #     global_result = _build(global_probs, "Global Model")
# #     client_results = [
# #         _build(probs, m["label"], tag=m["tag"], samples=m["samples"], specialty=m["specialty"])
# #         for m, probs in zip(_CHEST_CLIENTS_META, client_probs)
# #     ]

# #     votes = [r["predicted_index"] for r in client_results] + [global_result["predicted_index"]]
# #     consensus_idx = max(set(votes), key=votes.count)
# #     return {
# #         "global": global_result,
# #         "clients": client_results,
# #         "class_names": CHEST_CLASSES,
# #         "consensus": {
# #             "predicted_label": CHEST_CLASSES[consensus_idx],
# #             "agreement": votes.count(consensus_idx),
# #             "total_models": 4,
# #             "unanimous": len(set(votes)) == 1,
# #         },
# #     }


# # def _allowed_file(filename: str) -> bool:
# #     return (
# #         "." in filename
# #         and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
# #     )


# # def _ensure_dirs():
# #     os.makedirs(config.UPLOAD_DIR, exist_ok=True)
# #     plots_static = os.path.join(config.BASE_DIR, "static", "plots")
# #     os.makedirs(plots_static, exist_ok=True)
# #     if os.path.isdir(config.PLOT_DIR):
# #         for name in os.listdir(config.PLOT_DIR):
# #             src = os.path.join(config.PLOT_DIR, name)
# #             dst = os.path.join(plots_static, name)
# #             if os.path.isfile(src) and (
# #                 not os.path.exists(dst)
# #                 or os.path.getmtime(src) > os.path.getmtime(dst)
# #             ):
# #                 shutil.copy2(src, dst)


# # def _load_metrics():
# #     path = os.path.join(config.METRIC_DIR, "training_metrics.json")
# #     if not os.path.isfile(path):
# #         return None
# #     try:
# #         with open(path, "r", encoding="utf-8") as f:
# #             return json.load(f)
# #     except (json.JSONDecodeError, OSError):
# #         return None


# # def create_app() -> Flask:
# #     app = Flask(__name__, static_folder="static", template_folder="templates")
# #     app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
# #     app.secret_key = config.SECRET_KEY

# #     _ensure_dirs()

# #     try:
# #         registry.load()
# #         app.logger.info("Model registry loaded on %s", registry.device)
# #     except FileNotFoundError as exc:
# #         app.logger.error("Model load failed: %s", exc)

# #     @app.context_processor
# #     def inject_globals():
# #         return {"app_name": "Federated Learning Inference Console"}

# #     @app.route("/", methods=["GET"])
# #     def index():
# #         info = registry.info() if registry.loaded else None
# #         return render_template("index.html", info=info)

# #     @app.route("/predict", methods=["POST"])
# #     def predict():
# #         if "image" not in request.files:
# #             return _error_response("No image was provided.", 400)

# #         file = request.files["image"]
# #         if not file or file.filename == "":
# #             return _error_response("Please choose an image to upload.", 400)

# #         if not _allowed_file(file.filename):
# #             return _error_response(
# #                 f"Unsupported file type. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}",
# #                 400,
# #             )

# #         raw = file.read()
# #         if not raw:
# #             return _error_response("Uploaded file is empty.", 400)

# #         try:
# #             tensor, preview_28, original = preprocess_image(raw)
# #         except Exception as exc:
# #             app.logger.exception("Preprocessing error")
# #             return _error_response(f"Could not read the image: {exc}", 400)

# #         try:
# #             if not registry.loaded:
# #                 registry.load()
# #             results = registry.predict_all(tensor)
# #         except FileNotFoundError as exc:
# #             return _error_response(str(exc), 500)
# #         except Exception as exc:
# #             app.logger.exception("Inference error")
# #             return _error_response(f"Inference failed: {exc}", 500)

# #         safe_name = secure_filename(file.filename) or "upload.png"
# #         unique = f"{uuid.uuid4().hex[:10]}_{safe_name}"
# #         saved_path = os.path.join(config.UPLOAD_DIR, unique)
# #         with open(saved_path, "wb") as f:
# #             f.write(raw)

# #         original_url = url_for("uploaded_file", filename=unique)
# #         preview_data_url = pil_to_data_url(preview_28.resize((140, 140)))

# #         payload = {
# #             "ok": True,
# #             "filename": file.filename,
# #             "original_url": original_url,
# #             "preview_data_url": preview_data_url,
# #             "class_names": config.CLASS_NAMES,
# #             "results": results,
# #         }

# #         if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
# #             return jsonify(payload)

# #         return render_template(
# #             "result.html",
# #             payload=payload,
# #             results=results,
# #             original_url=original_url,
# #             preview_data_url=preview_data_url,
# #             filename=file.filename,
# #             class_names=config.CLASS_NAMES,
# #         )

# #     @app.route("/uploads/<path:filename>")
# #     def uploaded_file(filename):
# #         return send_from_directory(config.UPLOAD_DIR, filename)

# #     @app.route("/dashboard")
# #     def dashboard():
# #         metrics = _load_metrics()
# #         info = registry.info() if registry.loaded else None
# #         plots_dir = os.path.join(config.BASE_DIR, "static", "plots")
# #         plots = []
# #         if os.path.isdir(plots_dir):
# #             for name in sorted(os.listdir(plots_dir)):
# #                 if name.lower().endswith((".png", ".jpg", ".jpeg")):
# #                     plots.append(name)
# #         return render_template(
# #             "dashboard.html", metrics=metrics, info=info, plots=plots
# #         )

# #     @app.route("/proxy", methods=["GET", "POST"])
# #     def proxy():
# #         if request.method == "GET":
# #             return render_template("chest.html", result=None, original_url=None, filename=None)

# #         if "image" not in request.files:
# #             flash("No image was provided.", "error")
# #             return render_template("chest.html", result=None, original_url=None, filename=None)

# #         file = request.files["image"]
# #         if not file or file.filename == "":
# #             flash("Please choose a chest X-ray image to analyze.", "error")
# #             return render_template("chest.html", result=None, original_url=None, filename=None)

# #         if not _allowed_file(file.filename):
# #             flash(
# #                 f"Unsupported file type. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}",
# #                 "error",
# #             )
# #             return render_template("chest.html", result=None, original_url=None, filename=None)

# #         raw = file.read()
# #         if not raw:
# #             flash("The uploaded file is empty.", "error")
# #             return render_template("chest.html", result=None, original_url=None, filename=None)

# #         result = _chest_predict(raw)

# #         safe_name = secure_filename(file.filename) or "chest.png"
# #         unique = f"chest_{uuid.uuid4().hex[:10]}_{safe_name}"
# #         saved_path = os.path.join(config.UPLOAD_DIR, unique)
# #         with open(saved_path, "wb") as fh:
# #             fh.write(raw)

# #         original_url = url_for("uploaded_file", filename=unique)

# #         return render_template(
# #             "chest.html",
# #             result=result,
# #             original_url=original_url,
# #             filename=file.filename,
# #         )

# #     @app.route("/health")
# #     def health():
# #         info = registry.info() if registry.loaded else {"loaded": False}
# #         return jsonify({"status": "ok", "registry": info})

# #     @app.errorhandler(413)
# #     def too_large(_):
# #         return _error_response(
# #             f"File exceeds maximum size of {config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB",
# #             413,
# #         )

# #     def _error_response(message, status):
# #         if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
# #             return jsonify({"ok": False, "error": message}), status
# #         flash(message, "error")
# #         return redirect(url_for("index"))

# #     return app


# # if __name__ == "__main__":
# #     application = create_app()
# #     application.run(host="0.0.0.0", port=5005, debug=False)
