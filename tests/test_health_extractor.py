import numpy as np
import soundfile as sf

from voiceprint_features_144.extract_health_matrix import extract_health_matrix


def _make_test_wav(tmp_path, sr=16000, secs=0.5, freq=260.0):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False, dtype=np.float32)
    sig = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    wav_path = tmp_path / "health.wav"
    sf.write(str(wav_path), sig, sr)
    return wav_path


def test_health_matrix_shape_and_range(tmp_path):
    wav_path = _make_test_wav(tmp_path, sr=22050, secs=0.8, freq=310.0)

    target_frames = 96
    mat, sr, band = extract_health_matrix(
        str(wav_path),
        target_frames=target_frames,
        use_pcen=True,
        force_down_to_16k=True,
        fmin=120,
        fmax=4800,
    )

    assert mat.shape == (target_frames, 144)
    assert mat.dtype == np.uint8
    assert mat.min() >= 0 and mat.max() <= 255
    assert isinstance(sr, int)
    assert band == (120, 4800)
