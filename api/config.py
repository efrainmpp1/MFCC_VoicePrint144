import os

class Config:
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "8000"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(20 * 1024 * 1024)))  # 20 MB
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join("api", "uploads"))
    DEFAULT_MODE = os.getenv("DEFAULT_MODE", "mfcc")  # mfcc | logmel
    DEFAULT_PCEN = os.getenv("DEFAULT_PCEN", "0")     # "0" or "1"
    DEFAULT_DOWN16K = os.getenv("DEFAULT_DOWN16K", "1")  # "0" or "1"
    ALLOWED_EXTENSIONS = {"wav"}
