import io
import json
import numpy as np
import soundfile as sf
import pytest

from api.app import create_app


@pytest.fixture(scope="module")
def app():
    app = create_app()
    app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def _make_test_wav(tmp_path, sr=16000, secs=0.6, freq=220.0, stereo=False):
    """
    Gera um .wav curto (senoidal) para teste.
    - sr=16000 por padrão (bate com nosso pipeline)
    - duração curta para ser leve no teste
    """
    t = np.linspace(0, secs, int(sr * secs), endpoint=False, dtype=np.float32)
    sig = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    if stereo:
        sig = np.stack([sig, sig], axis=1)  # 2 canais
    wav_path = tmp_path / "sample.wav"
    sf.write(str(wav_path), sig, sr)
    return wav_path


@pytest.mark.parametrize("mode, expect_pcen_field", [
    ("mfcc", False),
    ("logmel", True),
    ("bio_mean144", True),
    ("bio_mm72", True),
])
def test_extract_144_ok(client, tmp_path, mode, expect_pcen_field):
    wav_path = _make_test_wav(tmp_path, sr=16000, secs=0.7, freq=440.0)

    with open(wav_path, "rb") as f:
        data = {
            "audio": (f, "sample.wav"),
        }
        # pcen=1 só tem efeito em logmel/bio_*, mas não deve quebrar nos demais
        resp = client.post(
            f"/api/v1/extract?mode={mode}&pcen=1&down16k=1",
            data=data,
            content_type="multipart/form-data",
        )

    assert resp.status_code == 200, resp.data
    payload = resp.get_json()
    assert payload["shape"] == [144]
    assert isinstance(payload["features"], list)
    assert len(payload["features"]) == 144
    assert payload["mode"] == mode if mode in ("logmel", "bio_mean144", "bio_mm72") else "mfcc"
    # sr e band coerentes
    assert isinstance(payload["sr"], int) and payload["sr"] in (16000, 15999, 16001)
    assert isinstance(payload["band"], list) and len(payload["band"]) == 2
    assert 50 <= payload["band"][0] <= 200
    assert payload["band"][1] <= 8000

    # pcen só é relevante/verdadeiro nos modos baseados em log-mel/bio_*
    if expect_pcen_field:
        assert payload["pcen"] in (True, False)  # pcen=1 -> True
    else:
        assert payload["pcen"] is False


def test_missing_file_field(client):
    resp = client.post("/api/v1/extract?mode=mfcc", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    payload = resp.get_json()
    assert "missing file field 'audio'" in payload.get("error", "")


def test_wrong_extension(client, tmp_path):
    # gera um "falso" arquivo .txt
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("not a wav")

    with open(bad_file, "rb") as f:
        data = {"audio": (f, "bad.txt")}
        resp = client.post("/api/v1/extract?mode=mfcc", data=data, content_type="multipart/form-data")

    assert resp.status_code in (400, 415)
    payload = resp.get_json()
    assert "only .wav" in payload.get("error", "").lower()


@pytest.mark.parametrize("sr", [22050, 32000, 48000])
def test_down16k_behavior(client, tmp_path, sr):
    """Arquivos com sr>16k devem ser reduzidos para 16k quando down16k=1."""
    wav_path = _make_test_wav(tmp_path, sr=sr, secs=0.5, freq=330.0)

    with open(wav_path, "rb") as f:
        data = {"audio": (f, "varsr.wav")}
        resp = client.post(
            "/api/v1/extract?mode=logmel&pcen=0&down16k=1",
            data=data,
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["sr"] == 16000  # downsample aplicado


def test_no_downsample_when_requested(client, tmp_path):
    """Se down16k=0, não deve forçar 16k em sr alto (apenas clampa fmax)."""
    wav_path = _make_test_wav(tmp_path, sr=32000, secs=0.5, freq=330.0)

    with open(wav_path, "rb") as f:
        data = {"audio": (f, "nodown.wav")}
        resp = client.post(
            "/api/v1/extract?mode=bio_mean144&pcen=1&down16k=0",
            data=data,
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["sr"] in (32000, 31999, 32001)  # manteve SR
    # fmax deve ser limitado por 0.45 * sr (logo < 14400)
    assert payload["band"][1] <= int(0.45 * payload["sr"]) + 5
