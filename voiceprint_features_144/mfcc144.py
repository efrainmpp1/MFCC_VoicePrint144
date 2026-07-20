import json
from typing import Tuple
import numpy as np
import soundfile as sf
import librosa
from .common_adaptive import to_mono, stft_params_from_sr, safe_voice_band

def _stats_mean_std(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1)
    sd = X.std(axis=1, ddof=1) if X.shape[1] > 1 else np.zeros(X.shape[0], dtype=np.float32)
    return np.concatenate([mu, sd], axis=0)

def _apply_vad_cms(
    M: np.ndarray,
    y: np.ndarray,
    n_fft: int,
    hop: int,
    top_db: float = 40.0,
    min_voiced: int = 3,
) -> np.ndarray:
    """
    VAD: descarta frames de silêncio (energia < top_db abaixo do pico).
    CMS: subtrai a média cepstral estimada sobre os frames voiced.
    Retorna M_cms com shape (n_mfcc, T_voiced).
    Fallback para M original se frames voiced < min_voiced.
    """
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]

    # Alinha tamanho com M (rms pode ter +1 frame por padding do librosa)
    T = M.shape[1]
    rms = rms[:T]

    rms_safe = np.maximum(rms, 1e-10)
    rms_db = 20.0 * np.log10(rms_safe / (rms_safe.max() + 1e-10))
    voiced_mask = rms_db > -top_db

    if voiced_mask.sum() < min_voiced:
        voiced_mask = np.ones(T, dtype=bool)

    M_voiced = M[:, voiced_mask]

    cms_mean = M_voiced.mean(axis=1, keepdims=True)
    return M_voiced - cms_mean

def extract_mfcc_144(
    wav_path: str,
    n_mfcc: int = 24,
    n_mels: int = 64,
    pre_emphasis: float = 0.97,
    force_down_to_16k: bool = True
) -> Tuple[np.ndarray, int, Tuple[int, int]]:
    """
    Lê um .wav e retorna:
      - features: vetor (144,) float32
      - sr: sample-rate efetiva
      - band: (fmin, fmax) usada na extração
    """
    y, sr = sf.read(wav_path, always_2d=False)
    y = to_mono(y).astype(np.float32)

    # Padroniza SR (opcional). Nunca upsample; apenas downsample se sr > 16k.
    if force_down_to_16k and sr > 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
        sr = 16000

    # Pré-ênfase ajuda em microfones de celular
    if len(y) > 1:
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    n_fft, hop = stft_params_from_sr(sr, 25.0, 10.0)
    fmin, fmax = safe_voice_band(sr, 100, 7200)

    M = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop, fmin=fmin, fmax=fmax, htk=True
    )  # (n_mfcc, T)

    M_cms = _apply_vad_cms(M, y, n_fft, hop)  # VAD + CMS → (n_mfcc, T_voiced)
    d1 = librosa.feature.delta(M_cms, order=1)
    d2 = librosa.feature.delta(M_cms, order=2)

    feat = np.concatenate([_stats_mean_std(M_cms), _stats_mean_std(d1), _stats_mean_std(d2)], axis=0).astype(np.float32)
    assert feat.shape[0] == n_mfcc * 3 * 2 == 144
    return feat, sr, (fmin, fmax)

if __name__ == "__main__":
    import sys
    vec, sr, band = extract_mfcc_144(sys.argv[1])
    print(json.dumps({"sr": int(sr), "band": band, "shape": [144], "features": vec.tolist()}))
