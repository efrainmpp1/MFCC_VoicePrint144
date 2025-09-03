# biometric144.py
import json
from typing import Tuple
import numpy as np
import soundfile as sf
import librosa
from .common_adaptive import to_mono, stft_params_from_sr, safe_voice_band

def _logmel(y, sr, n_bands: int, use_pcen: bool, fmin: int, fmax: int, n_fft: int, hop: int):
    # Espectrograma Mel (magnitude)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_bands, n_fft=n_fft, hop_length=hop,
        fmin=fmin, fmax=fmax, power=1.0
    )
    if use_pcen:
        X = librosa.pcen(S * (2**31), time_constant=0.06, eps=1e-6, power=0.25, gain=0.98, bias=2.0)
    else:
        X = librosa.power_to_db(S**2 + 1e-12, ref=np.max)
    return X  # shape: (n_bands, T)

def extract_biometric_144(
    wav_path: str,
    mode: str = "mean144",         # "mean144" (padrão) ou "mean_median_72"
    use_pcen: bool = False,
    force_down_to_16k: bool = True
) -> Tuple[np.ndarray, int, Tuple[int, int]]:
    """
    Extrai um vetor 144D sem derivadas e SEM variância/desvio:
      - mode="mean144": Log-Mel 144 bandas + média no tempo -> (144,)
      - mode="mean_median_72": Log-Mel 72 bandas + [média, mediana] -> (144,)
    Retorna: (features[144], sr, (fmin,fmax))
    """
    y, sr = sf.read(wav_path, always_2d=False)
    y = to_mono(y).astype(np.float32)

    # Downsample consistente (não faz upsample)
    if force_down_to_16k and sr > 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
        sr = 16000

    n_fft, hop = stft_params_from_sr(sr, 25.0, 10.0)
    fmin, fmax = safe_voice_band(sr, 100, 7200)

    if mode == "mean_median_72":
        X = _logmel(y, sr, n_bands=72, use_pcen=use_pcen, fmin=fmin, fmax=fmax, n_fft=n_fft, hop=hop)
        mean = X.mean(axis=1)
        med  = np.median(X, axis=1)
        feat = np.concatenate([mean, med], axis=0).astype(np.float32)  # 72*2 = 144
    else:
        # default: mean144
        X = _logmel(y, sr, n_bands=144, use_pcen=use_pcen, fmin=fmin, fmax=fmax, n_fft=n_fft, hop=hop)
        mean = X.mean(axis=1)
        feat = mean.astype(np.float32)  # 144*1 = 144

    assert feat.shape[0] == 144
    return feat, sr, (fmin, fmax)

if __name__ == "__main__":
    import sys
    vec, sr, band = extract_biometric_144(sys.argv[1], mode="mean144", use_pcen=False)
    print(json.dumps({"sr": int(sr), "band": band, "shape": [144], "features": vec.tolist()}))
