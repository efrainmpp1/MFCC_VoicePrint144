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
    fmax: int = 7000,
    vad_top_db: float = 40.0,
    min_voiced: int = 3,
) -> np.ndarray:
    """
    Retorna uma matriz (target_frames, 144), com valores normalizados por frame entre 0–255 (uint8).
    - 24 MFCCs + 24 Δ + 24 ΔΔ, duplicados por frame (total 144 features/frame)
    - VAD: descarta frames com energia < vad_top_db abaixo do pico antes de calcular features.
    - CMS: subtrai a média cepstral estimada sobre os frames voiced, removendo efeito de canal/microfone.
    - Frames silenciosas são representadas como zeros na matriz final (zero-pad após frames voiced).
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
    )  # (n_mfcc, T)

    # VAD: identifica frames voiced via energia RMS
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    T = M.shape[1]
    rms = rms[:T]
    rms_safe = np.maximum(rms, 1e-10)
    rms_db = 20.0 * np.log10(rms_safe / (rms_safe.max() + 1e-10))
    voiced_mask = rms_db > -vad_top_db
    if voiced_mask.sum() < min_voiced:
        voiced_mask = np.ones(T, dtype=bool)

    # CMS: subtrai média cepstral dos frames voiced — remove efeito de canal/microfone
    M_voiced = M[:, voiced_mask]  # (n_mfcc, T_voiced)
    M_cms = M_voiced - M_voiced.mean(axis=1, keepdims=True)

    d1 = librosa.feature.delta(M_cms, order=1)
    d2 = librosa.feature.delta(M_cms, order=2)

    full = np.concatenate([M_cms, d1, d2], axis=0).astype(np.float32)  # (72, T_voiced)
    full = full.T  # (T_voiced, 72)

    full = np.concatenate([full, full], axis=1)  # (T_voiced, 144)

    if full.shape[0] < target_frames:
        pad = np.zeros((target_frames - full.shape[0], full.shape[1]), dtype=np.float32)
        full = np.vstack([full, pad])
    elif full.shape[0] > target_frames:
        full = full[:target_frames, :]

    # Normaliza cada linha/frame para [0, 255] e converte para uint8 (vetorizado)
    row_min = full.min(axis=1, keepdims=True)
    row_max = full.max(axis=1, keepdims=True)
    row_range = row_max - row_min
    safe_range = np.where(row_range == 0, 1, row_range)

    norm = (full - row_min) / safe_range
    normalized = np.round(norm * 255).astype(np.uint8)
    normalized[(row_range == 0).squeeze(axis=1)] = 0

    return normalized, sr, (fmin, fmax)