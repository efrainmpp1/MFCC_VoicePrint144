# MFCC_VoicePrint144

Extract a fixed **144-dimensional audio feature vector** from any `.wav` file for **speaker biometrics and voice analysis**.
This project standardizes feature extraction using **MFCC + Δ + ΔΔ** or **Log-Mel (+PCEN)** within the human voice band (100–7200 Hz).
It provides three interfaces: **CLI**, **Python API**, and a **REST API (Flask)**.

---

## 📦 Features

- **MFCC 144D** → 24 MFCC × (static, Δ, ΔΔ) × (mean, std) = 144
- **Log-Mel 144D** → 48 Mel bands × (mean, std, median) = 144
- **MFCC matrix (frames × 144)** → MFCC/Δ/ΔΔ por quadro, duplicados para 144 colunas, com padding/clipping para `n_frames` e normalização 0–255
- **Health matrix (frames × 144)** → Log-Mel/PCEN por quadro (48 bandas) + deltas + energia + pitch, replicados até 144 colunas
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

# Health matrix (temporal, 144 cols/frame)
python -m voiceprint_features_144.cli examples/sample.wav --mode health_matrix --pcen --n-frames 256
```

### Options

- `--mode {mfcc|logmel|health_matrix}` → choose extractor (default: `mfcc`)
- `--pcen` → enable PCEN (for `logmel` or `health_matrix`)
- `--no-down16k` → do not downsample to 16 kHz when sr > 16k
- `--n-frames` / `--fmin` / `--fmax` → only for `health_matrix` (temporal output)
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

### Modes & query params

| Mode            | Output                                                              | Key params (query)                                                               |
| --------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `mfcc`          | `[144]` (MFCC + Δ + ΔΔ, mean/std)                                    | `down16k=0|1`                                                                    |
| `logmel`        | `[144]` (Log-Mel, mean/std/med)                                      | `pcen=0|1`, `down16k=0|1`                                                        |
| `bio_mean144`   | `[144]` (48 bandas, apenas média)                                    | `pcen=0|1`, `down16k=0|1`                                                        |
| `bio_mm72`      | `[144]` (72 bandas, média+mediana)                                   | `pcen=0|1`, `down16k=0|1`                                                        |
| `mfcc_matrix`   | `[n_frames, 144]` (MFCC/Δ/ΔΔ por quadro, normalizado 0–255)          | `n_frames` (default 20000), `fmin` (100), `fmax` (7000)                          |
| `health_matrix` | `[n_frames, 144]` (Log-Mel/PCEN + delta + energia + pitch, 0–255)    | `n_frames` (default 400), `fmin` (100), `fmax` (7200), `pcen=0|1`, `down16k=0|1` |
### Endpoints

- **Health check**

  ```
  GET /health
  → {"status": "ok"}
  ```

- **Feature extraction**

  ```
  POST /api/v1/extract?mode=mfcc|logmel|bio_mean144|bio_mm72|mfcc_matrix|health_matrix&pcen=0|1&down16k=0|1
  form-data: audio=@file.wav
  ```

Example request (with curl):

```bash
curl -X POST "http://localhost:8000/api/v1/extract?mode=logmel&pcen=1" \
  -F "audio=@examples/sample.wav"
```

Example request for temporal biometrics (`mfcc_matrix`):

```bash
curl -X POST "http://localhost:8000/api/v1/extract?mode=mfcc_matrix&n_frames=400&fmin=80&fmax=7200" \
  -F "audio=@examples/sample.wav"
```

Example request for health matrix (`health_matrix`, com PCEN):

```bash
curl -X POST "http://localhost:8000/api/v1/extract?mode=health_matrix&n_frames=256&fmin=120&fmax=4800&pcen=1" \
  -F "audio=@examples/sample.wav"
```

### Sobre o modo `health_matrix`

O `health_matrix` é um extrator temporal pensado para sensibilidade a variações de voz relacionadas a saúde (ex.: fadiga, rouquidão, gripe), mantendo 144 features por quadro e formato compatível com o pipeline de biometria.

- **O que ele calcula por quadro**
  - 48 bandas Log-Mel (ou PCEN se `pcen=1`)
  - 48 deltas de primeira ordem
  - Energia RMS (1 coluna)
  - Pitch estimado (1 coluna, em Hz)
  - As 98 colunas resultantes são replicadas/recortadas até 144 para manter consistência com outros modos.

- **Normalização e forma**
  - Cada linha é normalizada individualmente para o intervalo **0–255** (`uint8`).
  - A matriz final tem shape `[n_frames, 144]`, fazendo **padding** com zeros ou corte para atingir `n_frames`.

- **Parâmetros configuráveis (query ou CLI)**
  - `n_frames` (padrão `400`): total de quadros desejados na saída.
  - `fmin` / `fmax` (padrão `100` / `7200`): faixa de frequências passada ao banco Mel e estimativa de pitch (respeita o clamp de voz segura via `safe_voice_band`).
  - `pcen` (`0|1`, padrão `0`): ativa PCEN em vez de dB para maior robustez a variações de ganho.
  - `down16k` (`0|1`, padrão `1`): força downsample para 16 kHz quando o áudio estiver acima disso.

- **Quando usar**
  - Para treinar modelos temporais que avaliem variações de voz relacionadas a saúde ou estado vocal.
  - Para manter compatibilidade com o consumo já existente de matrizes 144D por quadro (sem alterar arquitetura downstream).

### Diferenças principais: `mfcc_matrix` vs. `health_matrix`
- **Propósito**
  - `mfcc_matrix`: biometria temporal, focado em MFCC/Δ/ΔΔ clássicos para reconhecimento de locutor.
  - `health_matrix`: sensibilidade a estado vocal/saúde, combinando Log-Mel/PCEN, energia e pitch.
- **Features por quadro**
  - `mfcc_matrix`: 24 MFCC + 24 Δ + 24 ΔΔ (72) duplicados até 144 colunas, todas cepstrais.
  - `health_matrix`: 48 Log-Mel/PCEN + 48 deltas + energia RMS + pitch (98) replicados/recortados até 144, misturando espectro, dinâmica e prosódia.
- **Configuração típica**
  - `mfcc_matrix`: `n_frames` padrão alto (20000), banda segura 100–7000 Hz, sem PCEN.
  - `health_matrix`: `n_frames` padrão moderado (400), banda 100–7200 Hz, PCEN opcional para robustez a ganho.
- **Resultado esperado**
  - Ambos retornam `[n_frames, 144]` normalizado 0–255 por linha, mas com objetivos diferentes: perfil cepstral para biometria (`mfcc_matrix`) versus variações vocais relacionadas a saúde (`health_matrix`).

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
- Use `mfcc_matrix` when you need the full sequência de MFCC/Δ/ΔΔ por quadro para modelos temporais de biometria.

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
