# Phyphox Vibration Mini-Lab (Hair Dryer) — Vibration + Acoustics Dashboard

A small “mini-lab” project demonstrating an end-to-end **condition monitoring workflow** using **Phyphox CSV exports** and **Python-based signal processing**.

- **Target demo:** Hair dryer as a rotating-machinery proxy (low vs high speed)
- **Outputs:** Repeatability checks, vibration + acoustics signatures, stakeholder-friendly summaries
- **Primary deliverable:** Streamlit dashboard (web app)

> ⚠️ Note: Phone sensors are not calibrated industrial instruments. This project focuses on methodology (measurement → repeatability → signature comparison → reporting), and documents practical limitations (sampling rate / Nyquist / potential aliasing).

---

## Repo Structure

```text
.
├─ app/                               # ✅ Main Streamlit web app
│  ├─ app_streamlit.py                # Streamlit dashboard entrypoint
│  └─ assets/                         # Branding assets used by the app
│     ├─ hero_dark.png
│     └─ logo_circle_dark.png
│
├─ .streamlit/                        # Streamlit configuration (theme/UI)
│  └─ config.toml
│
├─ phyphox_condition_monitoring.py    # Older CLI workflow (batch-style; base folder)
├─ analysis_results/                  # (optional) local outputs created by scripts
└─ README.md                          # This file
```

### What lives where?
- **`app/`**  
  The current “final” deliverable: an interactive **Streamlit dashboard** for non-technical stakeholders and engineers.
- **Repo root (base folder)**  
  Older / alternative workflows such as:
  - CLI scripts (run in terminal)
  - quick analysis utilities
  - one-off experiment scripts

---

## Getting Data (Phyphox)

### Vibration CSVs (required)
Use Phyphox:
- **Acceleration (without g)** or **Acceleration** (any axis or absolute acceleration)
- Record ~20–30 seconds per run
- Export as CSV

Recommended for repeatability:
- **2 runs at Low speed**
- **2 runs at High speed**

Suggested naming:
- `vib_low_run1.csv`, `vib_low_run2.csv`
- `vib_high_run1.csv`, `vib_high_run2.csv`

### Acoustics CSVs (optional)
Use **one type only** (don’t mix types in the same upload set):
- **Frequency history** (tone tracking) *OR*
- **Audio spectrum** (dominant tone)

Recommended:
- 1–2 runs Low + 1–2 runs High

Suggested naming:
- `audio_low_run1.csv`, `audio_high_run1.csv`

---

## Run the Streamlit App (Recommended)

### 1) Create & activate a virtual environment
**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
If you have a `requirements.txt`:
```bash
pip install -r requirements.txt
```

If not, minimum:
```bash
pip install streamlit numpy pandas scipy matplotlib
# optional (recommended for interactive charts):
pip install plotly
```

### 3) Run the app
From inside the `app/` folder:
```bash
cd app
streamlit run app_streamlit.py
```

The dashboard will open at:
- http://localhost:8501

### 4) Upload CSVs in the sidebar
- Upload vibration CSVs under **Low speed** and **High speed**
- (Optional) upload acoustics CSVs
- Use the **Stakeholder / Engineer** toggle for different views

---

## Streamlit Theme / Branding

Theme configuration:
- `.streamlit/config.toml`

Branding assets:
- `app/assets/hero_dark.png`
- `app/assets/logo_circle_dark.png`

If you change asset names, update the paths near the top of:
- `app/app_streamlit.py`

---

## Older / CLI Workflow (Base Folder)

The repo root contains an older “batch-style” Python workflow intended for terminal usage.  
Example script:
- `phyphox_condition_monitoring.py`

Typical usage (example):
```bash
python phyphox_condition_monitoring.py --vibration \
  --vib-low vib_low_run1.csv vib_low_run2.csv \
  --vib-high vib_high_run1.csv vib_high_run2.csv \
  --out analysis_results
```

This path is useful when you want:
- fully reproducible batch outputs
- easy automation
- no UI / browser

---

## Interpretation Notes (for interviews / documentation)

- **Repeatability:** needs ≥2 runs per speed. With only 1 run, repeatability metrics appear as 0 by definition.
- **Nyquist / aliasing:** phone sampling rate limits the highest measurable vibration frequency (≈ fs/2).
- **Industrial analogue:** real turbomachinery testing uses calibrated sensors + tach/keyphasor for speed reference.

---

## License
Add your preferred license here (MIT/Apache-2.0/etc.). If unsure, MIT is a common choice for portfolio projects.

---

## Contact
**Abhijith Sivaprasadan**  
LinkedIn: https://linkedin.com/in/typehelloworld/  
Email: abhijithsivaprasadan@gmail.com
