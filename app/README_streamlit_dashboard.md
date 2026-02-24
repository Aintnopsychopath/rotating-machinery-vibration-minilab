# Hair Dryer Condition Monitoring Dashboard (Streamlit)

## Install
```bash
pip install streamlit numpy pandas scipy matplotlib
# optional for better interactive charts:
pip install plotly
```

## Run
```bash
streamlit run app_streamlit.py
```

## Upload
### Vibration
Upload 2 low-speed vibration CSVs + 2 high-speed vibration CSVs (recommended).

### Acoustics (choose one type)
Either:
- Phyphox → Acoustics → Frequency history (CSV), or
- Phyphox → Acoustics → Audio spectrum (CSV)

Upload low + high (2 runs recommended).

## What stakeholders will see
- Simple takeaway (which speed vibrates more, tone shifts)
- Easy plots + downloadable summary tables
