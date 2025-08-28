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

    d1 = librosa.feature.delta(M, order=1)
    d2 = librosa.feature.delta(M, order=2)

    feat = np.concatenate([_stats_mean_std(M), _stats_mean_std(d1), _stats_mean_std(d2)], axis=0).astype(np.float32)
    assert feat.shape[0] == n_mfcc * 3 * 2 == 144
    return feat, sr, (fmin, fmax)

if __name__ == "__main__":
    import sys
    vec, sr, band = extract_mfcc_144(sys.argv[1])
    print(json.dumps({"sr": int(sr), "band": band, "shape": [144], "features": vec.tolist()}))
