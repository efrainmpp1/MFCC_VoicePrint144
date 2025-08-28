import json
from typing import Tuple
import numpy as np
import soundfile as sf
import librosa
from .common_adaptive import to_mono, stft_params_from_sr, safe_voice_band

def extract_logmel_144(
    wav_path: str,
    n_bands: int = 48,
    use_pcen: bool = False,
    force_down_to_16k: bool = True
) -> Tuple[np.ndarray, int, Tuple[int, int]]:
    """
    Lê um .wav e retorna:
      - features: vetor (144,) float32  [48 bandas × (mean,std,median)]
      - sr: sample-rate efetiva
      - band: (fmin, fmax) usada
    """
    y, sr = sf.read(wav_path, always_2d=False)
    y = to_mono(y).astype(np.float32)

    if force_down_to_16k and sr > 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
        sr = 16000

    n_fft, hop = stft_params_from_sr(sr, 25.0, 10.0)
    fmin, fmax = safe_voice_band(sr, 100, 7200)

    # Espectrograma Mel (magnitude)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_bands, n_fft=n_fft, hop_length=hop,
        fmin=fmin, fmax=fmax, power=1.0
    )  # (n_bands, T)

    if use_pcen:
        X = librosa.pcen(S * (2**31), time_constant=0.06, eps=1e-6, power=0.25, gain=0.98, bias=2.0)
    else:
        # Log-mel em dB (usa S**2 para energia e pequeno offset p/ estabilidade)
        X = librosa.power_to_db(S**2 + 1e-12, ref=np.max)

    mean = X.mean(axis=1)
    std  = X.std(axis=1, ddof=1) if X.shape[1] > 1 else np.zeros(X.shape[0], dtype=np.float32)
    med  = np.median(X, axis=1)

    feat = np.concatenate([mean, std, med], axis=0).astype(np.float32) 
    assert feat.shape[0] == 144
    return feat, sr, (fmin, fmax)

if __name__ == "__main__":
    import sys
    vec, sr, band = extract_logmel_144(sys.argv[1], use_pcen=False)
    print(json.dumps({"sr": int(sr), "band": band, "shape": [144], "features": vec.tolist()}))
