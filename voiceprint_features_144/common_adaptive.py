import numpy as np

def to_mono(y):
    return y if y.ndim == 1 else y.mean(axis=1)

def _next_pow2(n):
    return 1 << int(np.ceil(np.log2(max(1, n))))

def stft_params_from_sr(sr: int, win_ms: float = 25.0, hop_ms: float = 10.0):
    n_fft = _next_pow2(int(sr * (win_ms / 1000.0)))
    hop   = max(1, int(sr * (hop_ms / 1000.0)))
    return n_fft, hop

def safe_voice_band(sr: int, fmin: int = 100, fmax_safe: int = 7200):
    # clamp abaixo de Nyquist com margem
    fmax = min(fmax_safe, int(0.45 * sr))
    return fmin, fmax
