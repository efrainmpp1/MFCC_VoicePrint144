# MFCC_VoicePrint144

Extract a fixed **144-dimensional audio feature vector** from any `.wav` file for **speaker biometrics and voice analysis**.
This project standardizes feature extraction using **MFCC + Δ + ΔΔ** or **Log-Mel (+PCEN)** within the human voice band (100–7200 Hz).
It provides three interfaces: **CLI**, **Python API**, and a **REST API (Flask)**.

---

## 📦 Features

- **MFCC 144D** → 24 MFCC × (static, Δ, ΔΔ) × (mean, std) = 144
- **Log-Mel 144D** → 48 Mel bands × (mean, std, median) = 144
- **Adaptive STFT** (25 ms window / 10 ms hop, scaled to sample rate)
- **Safe frequency band:** 100–7200 Hz (clamped at 0.45 × sample rate)
- **Optional PCEN** for Log-Mel, robust to gain/recording differences
- **Consistent embeddings** across devices and sample rates
- **API-ready**: extract features via REST endpoint with form-data upload
- **Automated tests** with `pytest` for reliability

## 🎛️ Feature Extraction Parameters

| **Stage**           | **Parameter**          | **Value / Notes**                                            |
| ------------------- | ---------------------- | ------------------------------------------------------------ |
| **Pre-emphasis**    | Filter                 | `y[t] = x[t] – 0.97 × x[t-1]` (boosts high frequencies)      |
| **Framing**         | Window length          | 25 ms (\~400 samples @ 16 kHz)                               |
|                     | Hop length             | 10 ms (\~160 samples @ 16 kHz)                               |
| **FFT / STFT**      | FFT size (`n_fft`)     | 512 (adaptive to sample rate)                                |
| **Frequency range** | `fmin`                 | 100 Hz (cut-off below human voice)                           |
|                     | `fmax`                 | 7200 Hz (upper band for human voice, clamped at `0.45 × sr`) |
| **MFCC branch**     | # of MFCC coefficients | 24 (excluding 0th)                                           |
|                     | Δ (delta)              | 1st temporal derivative (captures dynamics)                  |
|                     | ΔΔ (delta-delta)       | 2nd temporal derivative (captures acceleration)              |
|                     | Statistics             | Mean + Std (per coef.)                                       |
|                     | Vector composition     | 24 × (static + Δ + ΔΔ) × 2 stats = **144D**                  |
| **Log-Mel branch**  | # of Mel bands         | 48                                                           |
|                     | PCEN (optional)        | Per-Channel Energy Normalization for robustness              |
|                     | Statistics             | Mean + Std + Median                                          |
|                     | Vector composition     | 48 × 3 stats = **144D**                                      |
| **Pooling**         | Method                 | Statistical pooling over all frames → fixed-length vector    |
| **Output**          | Shape                  | `[144]` (consistent across duration and device sample rate)  |

## 📂 Repository structure

```
MFCC_VoicePrint144/
├─ README.md
├─ requirements.txt
├─ examples/
│   └─ sample.wav
├─ voiceprint_features_144/
│   ├─ __init__.py
│   ├─ cli.py
│   ├─ common_adaptive.py
│   ├─ mfcc144.py
│   └─ mel144.py
├─ api/
│   ├─ __init__.py
│   ├─ app.py
│   ├─ config.py
│   ├─ wsgi.py
│   └─ uploads/
├─ tests/
│   ├─ test_api_extract.py
│   └─ test_feature_extractors.py
└─ .github/ (optional CI/CD workflows in future)
```

---

## ⚙️ Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install -e .
```

Dependencies:

- `numpy`
- `librosa`
- `soundfile`
- `resampy`
- `flask`
- `pytest` (for tests)

---

## 🖥️ Usage (CLI)

Run the extractor on a `.wav` file:

```bash
# MFCC 144D
python -m voiceprint_features_144.cli examples/sample.wav --mode mfcc

# Log-Mel 144D (with PCEN)
python -m voiceprint_features_144.cli examples/sample.wav --mode logmel --pcen
```

### Options

- `--mode {mfcc|logmel}` → choose extractor (default: `mfcc`)
- `--pcen` → enable PCEN (only for `logmel`)
- `--no-down16k` → do not downsample to 16 kHz when sr > 16k
- `--out file.json` → save JSON output

---

## 🐍 Usage (Python API)

```python
from voiceprint_features_144 import extract_mfcc_144, extract_logmel_144

# MFCC 144D
vec, sr, band = extract_mfcc_144("examples/sample.wav")
print(vec.shape)   # (144,)
print(sr, band)    # e.g., 16000, (100, 7200)

# Log-Mel 144D (with PCEN)
vec, sr, band = extract_logmel_144("examples/sample.wav", use_pcen=True)
```

Output example:

```json
{
  "sr": 16000,
  "band": [100, 7200],
  "mode": "mfcc",
  "shape": [144],
  "features": [ ... 144 floats ... ]
}
```

---

## 🌐 Usage (REST API)

Start the Flask server:

```bash
export FLASK_APP=api/wsgi.py
flask run --host=0.0.0.0 --port=8000
```

### Endpoints

- **Health check**

  ```
  GET /health
  → {"status": "ok"}
  ```

- **Feature extraction**

  ```
  POST /api/v1/extract?mode=mfcc|logmel&pcen=0|1&down16k=0|1
  form-data: audio=@file.wav
  ```

Example request (with curl):

```bash
curl -X POST "http://localhost:8000/api/v1/extract?mode=logmel&pcen=1" \
  -F "audio=@examples/sample.wav"
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

---

## 🔬 Notes

- If `sr < 16k`, no upsampling; `fmax` is clamped to `0.45*sr`.
- For `sr ≥ 16k`, audio is downsampled to 16k by default (configurable).
- Apply **z-score normalization** with training dataset statistics before NN usage.
- New mode (coming soon): **biometric-only extractor** (structural MFCCs without Δ/ΔΔ).

---

## 🧪 Running tests

Run all tests with:

```bash
pytest -q tests
```

Output (example):

```
..........                                                                                                           [100%]
10 passed in 1.76s
```

---

## 📜 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute this software, provided that the original license and copyright notice are included in all copies or substantial portions of the software.
See the [LICENSE](LICENSE) file for details.
