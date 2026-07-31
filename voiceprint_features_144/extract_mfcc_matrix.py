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
    fmax: int = 900
) -> np.ndarray:
    """
    Retorna uma matriz (target_frames, 144), com valores normalizados por
    coluna (por dimensão de feature, ao longo do tempo) entre 0-255 (uint8).
    - 23 MFCCs (c0/energia bruta descartado, escala de centenas dominava a
      normalização e esmagava as demais colunas quando normalizado por linha)
    - 24 Δ (inclui delta de c0)
    - 24 ΔΔ (inclui delta-delta de c0)
    Replicado/recortado até 144 features/frame.
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

    M_no_c0 = M[1:]  # descarta c0 (energia bruta absoluta)

    full = np.concatenate([M_no_c0, d1, d2], axis=0).astype(np.float32)  # (3*n_mfcc-1, T)
    full = full.T  # (T, 3*n_mfcc-1)

    n_features = full.shape[1]
    if n_features < 144:
        repeat_times = int(np.ceil(144 / n_features))
        full = np.tile(full, (1, repeat_times))[:, :144]
    elif n_features > 144:
        full = full[:, :144]

    # Trunca ANTES de normalizar, para o cálculo de min/max por coluna refletir
    # exatamente a janela de frames que será persistida.
    if full.shape[0] > target_frames:
        full = full[:target_frames, :]

    # Normaliza cada coluna (feature) ao longo do tempo para [0, 255].
    col_min = full.min(axis=0, keepdims=True)
    col_max = full.max(axis=0, keepdims=True)
    col_range = col_max - col_min
    safe_range = np.where(col_range == 0, 1, col_range)

    norm = (full - col_min) / safe_range
    normalized = np.round(norm * 255).astype(np.uint8)
    normalized[:, (col_range == 0).squeeze(axis=0)] = 0

    # Completa com os próprios frames reais repetidos ciclicamente (em vez de
    # zero-padding), feito SOMENTE depois de normalizar para não contaminar o
    # min/max real de cada coluna.
    if normalized.shape[0] < target_frames:
        repeat_times = int(np.ceil(target_frames / normalized.shape[0]))
        normalized = np.tile(normalized, (repeat_times, 1))[:target_frames, :]

    return normalized, sr, (fmin, fmax)