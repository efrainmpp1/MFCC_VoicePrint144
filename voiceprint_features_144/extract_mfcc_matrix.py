import numpy as np
import soundfile as sf
import librosa
from .common_adaptive import to_mono, stft_params_from_sr, safe_voice_band

def extract_mfcc_matrix(
    wav_path: str,
    n_mfcc: int = 24,
    n_mels: int = 64,
    target_frames: int = 20000,
    pre_emphasis: float = 0.97,
    force_down_to_16k: bool = True,
    fmin: int = 100,
    fmax: int = 7000
) -> np.ndarray:
    """
    Retorna uma matriz (target_frames, 144), sendo:
    - 24 MFCCs
    - 24 Δ
    - 24 ΔΔ
    Concatenados por linha/frame
    """
    y, sr = sf.read(wav_path, always_2d=False)
    y = to_mono(y).astype(np.float32)

    if force_down_to_16k and sr > 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
        sr = 16000

    if len(y) > 1:
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    n_fft, hop = stft_params_from_sr(sr, 25.0, 10.0)
    fmin, fmax = safe_voice_band(sr, fmin, fmax)

    M = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop, fmin=fmin, fmax=fmax, htk=True
    )
    d1 = librosa.feature.delta(M, order=1)
    d2 = librosa.feature.delta(M, order=2)

    full = np.concatenate([M, d1, d2], axis=0).astype(np.float32)  # (72, T)
    full = full.T  # (T, 72)

    # Repete para dar 144 por frame
    full = np.concatenate([full, full], axis=1)  # (T, 144)

    if full.shape[0] < target_frames:
        pad = np.zeros((target_frames - full.shape[0], full.shape[1]), dtype=np.float32)
        full = np.vstack([full, pad])
    elif full.shape[0] > target_frames:
        full = full[:target_frames, :]

    return full, sr, (fmin, fmax)
