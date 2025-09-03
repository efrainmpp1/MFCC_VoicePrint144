import os
import time
import uuid
from typing import Tuple, Dict, Any
from flask import Flask, request, jsonify
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from .config import Config
from voiceprint_features_144 import extract_mfcc_144, extract_logmel_144


# ---------- Helpers puros (reduzem complexidade da rota) ----------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def get_request_params() -> Tuple[str, bool, bool]:
    """Lê query params com defaults do Config e normaliza."""
    mode = (request.args.get("mode") or Config.DEFAULT_MODE).strip().lower()
    if mode not in ("mfcc", "logmel"):
        mode = "mfcc"
    pcen = (request.args.get("pcen") or Config.DEFAULT_PCEN) == "1"
    down16k = (request.args.get("down16k") or Config.DEFAULT_DOWN16K) == "1"
    return mode, pcen, down16k

def save_uploaded_wav(file: FileStorage, upload_dir: str) -> str:
    """Valida extensão e salva com nome único; retorna caminho salvo."""
    if file.filename == "":
        raise ValueError("empty filename")
    if not allowed_file(file.filename):
        raise ValueError("unsupported file type, only .wav allowed")

    os.makedirs(upload_dir, exist_ok=True)
    base = secure_filename(file.filename)
    ext = os.path.splitext(base)[1].lower() or ".wav"
    unique = f"{os.path.splitext(base)[0]}__{uuid.uuid4().hex}{ext}"
    path = os.path.join(upload_dir, unique)
    file.save(path)
    return path

def run_extractor(path: str, mode: str, pcen: bool, down16k: bool) -> Tuple[list, int, Tuple[int, int], str, bool]:
    """Executa o extrator escolhido e retorna (features, sr, band, mode_final, pcen_final)."""
    if mode == "logmel":
        vec, sr, band = extract_logmel_144(path, use_pcen=pcen, force_down_to_16k=down16k)
        return vec.tolist(), int(sr), (int(band[0]), int(band[1])), "logmel", bool(pcen)
    # default: mfcc
    vec, sr, band = extract_mfcc_144(path, force_down_to_16k=down16k)
    return vec.tolist(), int(sr), (int(band[0]), int(band[1])), "mfcc", False

def build_payload(features: list, sr: int, band: Tuple[int, int], mode: str, pcen: bool,
                  down16k: bool, latency_ms: int) -> Dict[str, Any]:
    return {
        "sr": sr,
        "band": [band[0], band[1]],
        "mode": mode,
        "pcen": pcen,
        "down16k": bool(down16k),
        "shape": [144],
        "features": [float(x) for x in features],
        "latency_ms": latency_ms,
    }


# ---------- App Factory (WSGI-friendly) ----------

def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.post("/api/v1/extract")
    def extract():
        """
        POST /api/v1/extract?mode=mfcc|logmel&pcen=0|1&down16k=0|1
        form-data: audio=@file.wav
        """
        t0 = time.time()

        # 1) parâmetros
        mode, pcen, down16k = get_request_params()

        # 2) arquivo (key obrigatória: 'audio')
        file = request.files.get("audio")
        if file is None:
            return jsonify({"error": "missing file field 'audio'"}), 400

        # 3) salvar temporário
        try:
            save_path = save_uploaded_wav(file, Config.UPLOAD_DIR)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            return jsonify({"error": f"failed to save file: {e}"}), 500

        # 4) extrair + montar payload
        try:
            vec, sr, band, mode_final, pcen_final = run_extractor(save_path, mode, pcen, down16k)
            latency = int((time.time() - t0) * 1000)
            return jsonify(build_payload(vec, sr, band, mode_final, pcen_final, down16k, latency)), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # 5) limpar arquivo
            try:
                os.remove(save_path)
            except Exception:
                pass

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": "file too large"}), 413

    return app
