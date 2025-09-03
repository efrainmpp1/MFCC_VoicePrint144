import os

class Config:
    # Host/Porta
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "8000"))

    # Uploads
    MAX_CONTENT_LENGTH = int(
        os.getenv("MAX_CONTENT_LENGTH", str(20 * 1024 * 1024))
    )  # 20 MB
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join("api", "uploads"))

    # Modos de extração disponíveis:
    # - mfcc: 24 MFCC + Δ + ΔΔ + stats -> 144D
    # - logmel: 48 bandas Log-Mel + (média, desvio, mediana) -> 144D
    # - bio_mean144: 144 bandas Log-Mel + média -> 144D (estrutural puro, sem variância temporal)
    # - bio_mm72: 72 bandas Log-Mel + (média, mediana) -> 144D (estrutural robusto, sem variância)
    DEFAULT_MODE = os.getenv("DEFAULT_MODE", "mfcc")  

    # Param extra: PCEN (apenas para logmel e modos bio_*)
    DEFAULT_PCEN = os.getenv("DEFAULT_PCEN", "0")     # "0" ou "1"

    # Downsample para 16kHz se sr > 16k
    DEFAULT_DOWN16K = os.getenv("DEFAULT_DOWN16K", "1")  # "0" ou "1"

    # Extensões permitidas
    ALLOWED_EXTENSIONS = {"wav"}
