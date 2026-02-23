# Phyphox Vibration Mini‑Lab (Hair Dryer → Turbine‑style Workflow)

A small **one‑day** project demonstrating an end‑to‑end vibration workflow:

**sensor mounting → data acquisition (Phyphox) → filtering → FFT/PSD/spectrogram → comparison of operating points → report**

This is intentionally a *low-cost proxy* for real rotating‑machinery diagnostics (fans, pumps, and—in principle—the same logic used when interpreting turbine vibration signals).

## What you get (outputs)
Running the script generates a folder `phyphox_full_report/` containing:
- a plain‑English `report.html` (shareable with non‑engineers)
- `summary_metrics.csv` (numbers in spreadsheet form)
- plots:
  - raw & filtered time traces
  - rolling RMS (“vibration strength” over time)
  - FFT & PSD overlays (frequency content)
  - spectrograms (how frequencies evolve over time)
  - a simple bar chart comparing key metrics

A sample output is included in:
- `docs/sample-output/`

## Reproduce the experiment (10 minutes)
### 1) Record in Phyphox
1. Choose **Raw Sensors → Accelerometer** (or **Acceleration with g**).
2. Securely tape the phone to the device (hair dryer body). **Do not let it move.**
3. Record ~30 seconds at one setting (e.g., low).
4. Record ~30 seconds at another setting (e.g., max).
5. Export each run as **CSV**.

### 2) Place CSVs into `data/`
Recommended filenames:
- `data/Raw Data Speed 1.csv`
- `data/Raw Data Speed 2.csv`

> By default this repo does **not** commit raw CSV data (`.gitignore`), to keep the repo clean.

## Run the analysis
### Option A: Python (recommended)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python src/phyphox_all_in_one.py
```

Then open:
- `phyphox_full_report/report.html`

## How to explain this to non‑engineers
Use the report’s “Quick numbers” section and this one‑liner:

> “We recorded how much the device shakes, then converted the signal into a ‘frequency fingerprint’ (FFT/PSD) to see which vibration tone dominates and how it changes between low and high speed.”

## Engineering note (important): Nyquist / aliasing
Phone sensors typically sample around a few hundred Hz.
That means the **highest reliable frequency is ~fs/2 (Nyquist)**.

If the *true* rotation-related frequency is above Nyquist, it may show up as a **mirrored (aliased) peak**:
`f_true ≈ fs − f_observed`

Industrial turbine diagnostics usually include:
- calibrated piezo sensors
- validated measurement chains
- tachometer / keyphasor (so “1X” order tracking is unambiguous)

This project still demonstrates the *workflow mindset* that’s relevant to rotating machinery troubleshooting.

## Repository layout
```text
.
├─ src/
│  └─ phyphox_all_in_one.py
├─ docs/
│  └─ sample-output/
├─ data/               # keep local
├─ RESULTS.md
├─ BRANCHING.md
└─ requirements.txt
```

## Suggested “runs as branches” approach
See `BRANCHING.md` for a simple workflow:
- `main` = code + docs
- `runs/<device>-<YYYY-MM-DD>` = snapshot with outputs for a particular experiment

## License
MIT — see `LICENSE`.
