import numpy as np
import soundfile as sf
import librosa

from .common_adaptive import to_mono, stft_params_from_sr, safe_voice_band


def _normalize_row_to_uint8(row: np.ndarray) -> np.ndarray:
    """Normaliza um vetor 1D para [0, 255] em uint8."""
    finite = row[np.isfinite(row)]
    if finite.size == 0:
        return np.zeros_like(row, dtype=np.uint8)
    min_val = finite.min()
    max_val = finite.max()
    if max_val - min_val == 0:
        return np.zeros_like(row, dtype=np.uint8)
    norm = (row - min_val) / (max_val - min_val)
    return np.round(norm * 255).astype(np.uint8)


def extract_health_matrix(
    wav_path: str,
    n_mels: int = 48,
    target_frames: int = 400,
    pre_emphasis: float = 0.97,
    use_pcen: bool = False,
    force_down_to_16k: bool = True,
    fmin: int = 100,
    fmax: int = 7200,
) -> np.ndarray:
    """
    Extrai uma matriz (target_frames, 144) sensível a variações de saúde vocal.

    Por quadro, concatena:
      - 48 bandas Mel (PCEN ou dB)
      - 48 deltas de primeira ordem
      - energia RMS (1 coluna)
      - pitch estimado (1 coluna, em Hz)
    O conjunto (98 colunas) é replicado/recortado até 144 colunas
    e cada linha é normalizada para [0, 255] (uint8).
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

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=1.0,
    )

    if use_pcen:
        base = librosa.pcen(mel, time_constant=0.06, eps=1e-6, b=0.5)
    else:
        base = librosa.power_to_db(mel, ref=np.max)

    d1 = librosa.feature.delta(base, order=1)

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop, center=False)[0]  # (T,)
    pitch = librosa.yin(y, fmin=fmin, fmax=min(fmax, sr // 2 - 1), sr=sr, frame_length=n_fft, hop_length=hop)

    # Transpõe para shape (T, n_mels) e concatena colunas adicionais
    base_t = base.T.astype(np.float32)
    d1_t = d1.T.astype(np.float32)
    energy_col = rms.reshape(-1, 1).astype(np.float32)
    pitch_med = float(np.nanmedian(pitch)) if np.isfinite(pitch).any() else 0.0
    pitch_col = np.where(np.isfinite(pitch), pitch, pitch_med).reshape(-1, 1).astype(np.float32)

    min_len = min(base_t.shape[0], d1_t.shape[0], energy_col.shape[0], pitch_col.shape[0])
    base_t = base_t[:min_len]
    d1_t = d1_t[:min_len]
    energy_col = energy_col[:min_len]
    pitch_col = pitch_col[:min_len]

    full = np.concatenate([base_t, d1_t, energy_col, pitch_col], axis=1)  # (T, 98)

    # Sanitiza NaNs/Infs antes da normalização por linha
    full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)

    # Replica colunas para atingir 144 features/frame
    if full.shape[1] < 144:
        repeat_times = int(np.ceil(144 / full.shape[1]))
        tiled = np.tile(full, (1, repeat_times))
        full = tiled[:, :144]
    elif full.shape[1] > 144:
        full = full[:, :144]

    # Ajusta número de frames
    if full.shape[0] < target_frames:
        pad = np.zeros((target_frames - full.shape[0], full.shape[1]), dtype=np.float32)
        full = np.vstack([full, pad])
    elif full.shape[0] > target_frames:
        full = full[:target_frames, :]

    normalized = np.zeros_like(full, dtype=np.uint8)
    for i in range(full.shape[0]):
        normalized[i] = _normalize_row_to_uint8(full[i])

    return normalized, sr, (fmin, fmax)
