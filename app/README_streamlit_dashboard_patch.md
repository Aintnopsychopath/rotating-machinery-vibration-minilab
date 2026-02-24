# Streamlit Dashboard (Fixed + Prettier)

## Install
```bash
pip install streamlit numpy pandas scipy matplotlib
# optional (recommended) for interactive zoom/hover plots:
pip install plotly
```

## Run
```bash
streamlit run app_streamlit.py
```

## Why you got NameError before
The dynamic 'plain English' summary must be computed only after the low/high summaries exist.
This version computes summaries first and calls the explainer function only when data is ready.
