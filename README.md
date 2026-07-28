# MFCC_VoicePrint144

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-0.1.0-informational.svg)](pyproject.toml)

Fixed-size **144-dimensional audio feature extraction** for speaker biometrics and
voice-health analysis. Every extractor in this library — whether it returns a single
pooled vector or a full per-frame matrix — always produces exactly **144 features**,
so downstream consumers never have to branch on feature dimensionality.

The library is exposed through three interfaces:

- a **Python API** (import and call directly),
- a **CLI** (`python -m voiceprint_features_144.cli`, or `vw-extract` once installed),
- and a **Flask REST API** (used in production as the `mfcc_extractor` Docker service
  consumed by the `voicewaves-backend` Node application).

---

## Table of contents

- [Extraction modes](#extraction-modes)
- [`mfcc_matrix` and `health_matrix` in detail](#mfcc_matrix-and-health_matrix-in-detail)
- [Business rules](#business-rules)
- [Installation](#installation)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Docker](#docker)
- [License](#license)

---

## Extraction modes

Six extraction modes are available, selected via `mode=` (API) or `--mode` (CLI).
All of them are adaptive to sample rate (25 ms window / 10 ms hop, scaled to `sr`)
and clamp their frequency band to a safe voice range via `safe_voice_band`
(`fmin`/`fmax`, capped at `0.45 × sr`).

| Mode | Shape | Output type | Signal processing | Purpose |
| --- | --- | --- | --- | --- |
| `mfcc` | `[144]` | mean+std pooled vector | 24 MFCC, **VAD** (−40 dB RMS gate) + **CMS** (cepstral mean subtraction over voiced frames), Δ, ΔΔ | Speaker identity, VAD/CMS-robust to silence and channel bias |
| `logmel` | `[144]` | mean+std+median pooled vector | 48-band Log-Mel or PCEN, no VAD/delta | Lightweight spectral-shape fingerprint |
| `bio_mean144` | `[144]` | mean-only pooled vector | 144-band Log-Mel/PCEN | Pure structural signature, no temporal variance |
| `bio_mm72` | `[144]` | mean+median pooled vector | 72-band Log-Mel/PCEN | Structural signature, robust to outlier frames |
| `mfcc_matrix` | `[target_frames, 144]` | per-frame matrix, `uint8` 0–255 | 23 MFCC (c0 dropped) + Δ + ΔΔ | Biometric identity (AS/perfil creation) |
| `health_matrix` | `[target_frames, 144]` | per-frame matrix, `uint8` 0–255 | 48-band Mel/PCEN + Δ + RMS energy + pitch (YIN) | Vocal-state / health modulation tracking |

`mfcc`, `logmel`, `bio_mean144` and `bio_mm72` collapse the whole clip into a single
144-value vector via statistical pooling. `mfcc_matrix` and `health_matrix` instead
preserve the temporal axis — one row per 10 ms frame — which is why they need the
extra normalization and padding rules described below.

---

## `mfcc_matrix` and `health_matrix` in detail

These two modes are the ones actively used by the `voicewaves-backend` production
pipeline, and the ones this document tracks most closely as they evolve.

### Column composition

| | `mfcc_matrix` | `health_matrix` |
| --- | --- | --- |
| Real columns before tiling | 71 | 98 |
| Composition | 23 MFCC (**c0 dropped**) + 24 Δ (incl. Δ of c0) + 24 ΔΔ (incl. ΔΔ of c0) | 48 Mel/PCEN bands + 48 Δ + 1 RMS energy + 1 pitch (Hz, via YIN) |
| Tiled/cropped to | 144 columns | 144 columns |
| Default `target_frames` | 20000 | 400 |
| Default `fmin` / `fmax` | 100 / 7000 Hz | 100 / 7200 Hz |

`c0` — the raw MFCC log-energy coefficient — is deliberately excluded from
`mfcc_matrix`'s feature set. Its absolute scale (hundreds) is an order of magnitude
larger than every other MFCC coefficient and delta (tens), and highly sensitive to
microphone gain rather than speaker identity. Its Δ and ΔΔ are still included, since
first/second-order dynamics of energy remain informative and are far smaller in
scale.

### Normalization: per column, not per row

Each of the 144 output columns is **min-max normalized to `[0, 255]` independently**,
using only the clip's real (non-padded) frames:

```
normalized[:, j] = round((raw[:, j] - min(raw[:, j])) / (max(raw[:, j]) - min(raw[:, j])) * 255)
```

This replaced an earlier per-row normalization scheme. Per-row normalization let
whichever feature had the largest absolute scale in a given frame — raw MFCC `c0` in
`mfcc_matrix`, or pitch in Hz in `health_matrix` — dominate that row's min/max range,
compressing every other feature's real variation into a narrow band near 255 and
destroying most of its resolution once quantized to `uint8`. Per-column normalization
gives every feature its own dynamic range, so no single feature can crush another's.

As a side effect, per-column min-max normalization is invariant to a constant
additive offset within a single recording — so a fixed microphone/channel bias on a
given feature (assuming it stays constant for the duration of one clip) cancels out
automatically. It does **not** filter out background noise or make dynamic ranges
directly comparable *across* recordings made under different noise conditions.

### Frame count: cyclic repetition instead of zero-padding

Both extractors must always return exactly `[target_frames, 144]`. What happens when
the real audio doesn't have that many frames — the common case, since typical
recordings here are ~5 seconds (≈500–600 frames of real signal) against a
`target_frames` of 20000 for `mfcc_matrix`:

- **Real frames < `target_frames`** (the common, short-clip case): the real frames
  are normalized first, then **tiled cyclically** (`np.tile` + truncate) to fill the
  remaining rows. The real signal repeats until the matrix is full — there are no
  all-zero filler rows.
- **Real frames > `target_frames`**: the matrix is truncated to the first
  `target_frames` rows before normalization. No repetition happens in this
  direction.

This replaced an earlier scheme that zero-padded short clips — for a 5-second clip
against `target_frames=20000`, that meant well over 95% of every output matrix was
literally zero. Zero-padding also risked skewing per-column min/max if it had been
computed after padding; normalization here always runs on the real window first, so
padding (whichever form) never affects the real values' scale.

---

## Business rules

- Output shape is always exactly `[target_frames, 144]`, dtype `uint8`, values in
  `0–255` — no exceptions, regardless of clip length.
- Normalization and frame-count adjustment always operate on the real captured
  window only; padding never influences the real values' scale.
- `mfcc_matrix` output feeds AS/perfil biometric **identity** creation in
  `voicewaves-backend`; `health_matrix` output feeds the "modulação" vocal-state
  **comparison** pipeline. The two are not interchangeable — a `health_matrix` file
  cannot be consumed where an `mfcc_matrix` file is expected, or vice versa.
- Files generated by the current normalization scheme (per-column, cyclic-fill) are
  **not directly value-comparable** to files generated by the older scheme (per-row,
  zero-padded). Any comparison or centroid computation mixing the two eras will
  produce misleading results.

---

## Installation

Requires **Python 3.12+**. System dependencies `ffmpeg` and `libsndfile1` are needed
for audio decoding (`soundfile`/`librosa`) and for `.ogg`→`.wav` batch conversion
(see [`scripts/`](#usage)).

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install -e .
```

Core dependencies: `numpy`, `librosa`, `soundfile`, `resampy`, `flask`, `gunicorn`
(production server), `pytest` (tests). See [`requirements.txt`](requirements.txt) for
pinned versions.

---

## Usage

### Python API

```python
from voiceprint_features_144 import extract_mfcc_144, extract_logmel_144
from voiceprint_features_144.extract_mfcc_matrix import extract_mfcc_matrix
from voiceprint_features_144.extract_health_matrix import extract_health_matrix

# Pooled 144D vector
vec, sr, band = extract_mfcc_144("path/to/audio.wav")
print(vec.shape)  # (144,)

# Per-frame biometric identity matrix
matrix, sr, band = extract_mfcc_matrix("path/to/audio.wav", target_frames=20000)
print(matrix.shape, matrix.dtype)  # (20000, 144) uint8

# Per-frame vocal-health matrix
matrix, sr, band = extract_health_matrix("path/to/audio.wav", target_frames=400)
print(matrix.shape, matrix.dtype)  # (400, 144) uint8
```

### CLI

```bash
python -m voiceprint_features_144.cli path/to/audio.wav --mode mfcc
python -m voiceprint_features_144.cli path/to/audio.wav --mode logmel --pcen
python -m voiceprint_features_144.cli path/to/audio.wav --mode health_matrix --n-frames 400 --pcen
```

The console script `vw-extract` (installed via `pip install -e .`) is an equivalent
shortcut for the same CLI. Only `mfcc`, `logmel` and `health_matrix` are available
through the CLI; `mfcc_matrix` (and `bio_mean144`/`bio_mm72`) are reachable via the
Python API and the REST API described below.

**Options**: `--mode {mfcc|logmel|health_matrix}` (default `mfcc`) · `--pcen`
(enable PCEN for `logmel`/`health_matrix`) · `--no-down16k` (skip forced 16 kHz
downsampling) · `--n-frames` / `--fmin` / `--fmax` (`health_matrix` only) · `--out
file.json` (write JSON to disk instead of stdout).

### REST API

Start the server:

```bash
export FLASK_APP=api/wsgi.py
flask run --host=0.0.0.0 --port=8000
```

In production this runs as the `mfcc_extractor` Docker service (see
[Docker](#docker)), called over HTTP by `voicewaves-backend`'s
`AudioFeatureExtractionService`.

**Endpoints**

```
GET  /health
POST /api/v1/extract?mode=mfcc|logmel|bio_mean144|bio_mm72|mfcc_matrix|health_matrix
     form-data: file=@audio.wav
```

Query params by mode:

| Mode | Params |
| --- | --- |
| `mfcc` | `down16k=0\|1` |
| `logmel` | `pcen=0\|1`, `down16k=0\|1` |
| `bio_mean144`, `bio_mm72` | `pcen=0\|1`, `down16k=0\|1` |
| `mfcc_matrix` | `n_frames` (default `20000`), `fmin` (`100`), `fmax` (`7000`) |
| `health_matrix` | `n_frames` (default `400`), `fmin` (`100`), `fmax` (`7200`), `pcen=0\|1`, `down16k=0\|1` |

Example requests:

```bash
curl -X POST "http://localhost:8000/api/v1/extract?mode=logmel&pcen=1" \
  -F "file=@path/to/audio.wav"

curl -X POST "http://localhost:8000/api/v1/extract?mode=mfcc_matrix&n_frames=20000&fmin=100&fmax=7000" \
  -F "file=@path/to/audio.wav"

curl -X POST "http://localhost:8000/api/v1/extract?mode=health_matrix&n_frames=400&pcen=1" \
  -F "file=@path/to/audio.wav"
```

Example response:

```json
{
  "sr": 16000,
  "band": [100, 7200],
  "mode": "logmel",
  "pcen": true,
  "down16k": true,
  "shape": [144],
  "features": [ ... 144 floats ... ],
  "latency_ms": 42
}
```

### Batch scripts

Two standalone scripts under [`scripts/`](scripts/) run extraction locally over a
folder of files, without going through the HTTP API — used to validate changes
against real datasets in [`examples/`](examples/).

- **`batch_wav_to_txtgz.py`** — walks a folder of `.wav` files, runs
  `extract_mfcc_matrix` on each, and writes gzip-compressed, whitespace-separated
  `.txt.gz` matrices to an output folder, mirroring the input's subfolder structure.

  ```bash
  python scripts/batch_wav_to_txtgz.py path/to/wav_folder --out path/to/output --n-frames 20000
  ```

- **`batch_ogg_to_health_txtgz.py`** — converts `.ogg` voice notes (e.g. WhatsApp
  PTT recordings) to `.wav` via `ffmpeg` (mono, 44100 Hz, PCM16 — matching
  `voicewaves-backend`'s own audio-conversion parameters), then runs
  `extract_health_matrix` on each and writes `.txt.gz` matrices, mirroring the
  input's subfolder structure (e.g. one subfolder per labeled vocal state).

  ```bash
  python scripts/batch_ogg_to_health_txtgz.py path/to/ogg_folder --out path/to/output --n-frames 20000
  ```

---

## Project structure

```
MFCC_VoicePrint144/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── api/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── wsgi.py
│   └── uploads/            # created at runtime, not checked in
├── voiceprint_features_144/
│   ├── __init__.py
│   ├── cli.py
│   ├── common_adaptive.py     # shared STFT/band-safety helpers
│   ├── mfcc144.py              # `mfcc` mode
│   ├── mel144.py                # `logmel` mode
│   ├── biometric144.py         # `bio_mean144` / `bio_mm72` modes
│   ├── extract_mfcc_matrix.py  # `mfcc_matrix` mode
│   └── extract_health_matrix.py # `health_matrix` mode
├── scripts/
│   ├── batch_wav_to_txtgz.py
│   └── batch_ogg_to_health_txtgz.py
├── tests/
│   ├── conftest.py
│   ├── test_api_extract.py
│   └── test_health_extractor.py
└── examples/                # local datasets used for manual validation
```

---

## Testing

```bash
pytest -q tests
```

- **`test_api_extract.py`** — Flask test-client integration tests against
  `/api/v1/extract`: happy paths for all six modes, missing/invalid file handling,
  sample-rate downsampling behavior, and shape/param checks for `mfcc_matrix` and
  `health_matrix`.
- **`test_health_extractor.py`** — unit tests directly against
  `extract_health_matrix`: output shape/dtype/value-range checks and robustness to
  degenerate pitch input (silence/noise).

---

## Docker

The Flask API is packaged as a standalone service (`Dockerfile`, base image
`python:3.12-slim`) that installs `ffmpeg` and `libsndfile1`, then serves via
`gunicorn` on the port defined by the `FLASK_PORT` env var (default `8000`). In
`voicewaves-backend`'s `docker-compose.yml`, this image is built as the
`mfcc_extractor` service and consumed exclusively over HTTP — no other service
imports this repository's Python code directly.

---

## License

MIT — see [LICENSE](LICENSE) for the full text.
