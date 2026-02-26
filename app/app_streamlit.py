# app_streamlit.py
# Hair Dryer Condition Monitoring — Vibration + Acoustics (Phyphox)
#
# Install:
#   pip install streamlit numpy pandas scipy matplotlib
#   pip install plotly   # optional (recommended) for zoom/hover plots
#
# Run:
#   cd app
#   streamlit run app_streamlit.py

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy import signal as sp_signal
import matplotlib.pyplot as plt

# ---------- Optional Plotly ----------
PLOTLY_OK = False
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ---------- MUST be first Streamlit call ----------
st.set_page_config(page_title="Hair Dryer Vibration + Acoustics", layout="wide")

# ---------- Paths (robust) ----------
APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"
HERO = ASSETS / "hero_dark.png"
LOGO = ASSETS / "logo_circle_dark.png"

# ---------- Dark UI CSS ----------
CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1200px; }
.small { color: rgba(229,231,235,0.72); font-size: 0.95rem; }
.card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.04);
  box-shadow: 0 12px 28px rgba(0,0,0,0.28);
  margin-bottom: 14px;
}
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(99,102,241,0.14);
  border: 1px solid rgba(99,102,241,0.32);
  color: rgba(224,231,255,0.95);
  font-size: 0.85rem;
}
.badge2 {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(16,185,129,0.14);
  border: 1px solid rgba(16,185,129,0.30);
  color: rgba(209,250,229,0.95);
  font-size: 0.85rem;
}
.hr {
  height: 1px;
  background: rgba(255,255,255,0.10);
  border: 0;
  margin: 10px 0 14px 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

if HERO.exists():
    st.image(str(HERO), width="stretch")
else:
    st.warning(f"Hero banner not found at: {HERO}. Put hero_dark.png under app/assets/")

with st.sidebar:
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if LOGO.exists():
        st.image(str(LOGO), width=120)
    else:
        st.warning(f"Logo not found at: {LOGO}. Put logo_circle_dark.png under app/assets/")

# ---------- Helpers ----------
def _find_col(cols, contains_any, contains_all=None):
    cols = list(cols)
    low = [c.lower() for c in cols]
    for c, cl in zip(cols, low):
        ok_any = any(s.lower() in cl for s in contains_any)
        ok_all = True if not contains_all else all(s.lower() in cl for s in contains_all)
        if ok_any and ok_all:
            return c
    return None

def _read_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def _safe_label(file_name: str, idx: int) -> str:
    # Handles repeated names like "Raw Data.csv"
    return f"{file_name} (run {idx+1})"

def _iqr(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    return float(q3 - q1)

# ---------- Vibration processing ----------
def load_vibration_df(df: pd.DataFrame):
    tcol = _find_col(df.columns, ["time"], ["s"]) or _find_col(df.columns, ["time"])
    if tcol is None:
        raise ValueError(f"Could not find a time column. Columns: {list(df.columns)}")

    acol = (_find_col(df.columns, ["absolute acceleration"], ["m/s"]) or
            _find_col(df.columns, ["absolute acceleration"]) or
            _find_col(df.columns, ["linear acceleration"], ["m/s"]) or
            _find_col(df.columns, ["acceleration"], ["m/s"]) or
            _find_col(df.columns, ["acceleration"]))
    if acol is None:
        raise ValueError(f"Could not find an acceleration column. Columns: {list(df.columns)}")

    t = df[tcol].to_numpy(dtype=float)
    x = df[acol].to_numpy(dtype=float)
    return t, x

def resample_uniform(t, x):
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    tu = np.arange(t[0], t[-1], dt)
    xu = np.interp(tu, t, x)
    return tu, xu, fs

def preprocess_vibration(x, fs, hp=5.0, lp=200.0):
    y = sp_signal.detrend(x, type="linear")
    nyq = 0.5 * fs
    lo = hp / nyq
    hi = min(lp / nyq, 0.999)
    if hi <= lo:
        return y
    b, a = sp_signal.butter(4, [lo, hi], btype="bandpass")
    return sp_signal.filtfilt(b, a, y)

def welch_psd(y, fs):
    nperseg = min(2048, max(256, len(y)//4))
    f, pxx = sp_signal.welch(y, fs=fs, nperseg=nperseg, scaling="density")
    return f, pxx

def dominant_freq_from_psd(f, pxx, fmin=10, fmax=200):
    band = (f >= fmin) & (f <= fmax)
    fb, sb = f[band], pxx[band]
    i = int(np.argmax(sb))
    return float(fb[i])

def spectrogram_db(y, fs):
    f, t, S = sp_signal.spectrogram(
        y, fs=fs,
        nperseg=min(1024, max(128, len(y)//6)),
        scaling="density", mode="psd"
    )
    SdB = 10*np.log10(S + 1e-20)
    return f, t, SdB

def analyze_vibration_file(uploaded_file, hp, lp, fmin, fmax, label):
    df = _read_csv(uploaded_file)
    t_raw, x_raw = load_vibration_df(df)
    t, x, fs = resample_uniform(t_raw, x_raw)
    y = preprocess_vibration(x, fs, hp=hp, lp=lp)

    f, pxx = welch_psd(y, fs)
    nyq = 0.5 * fs
    fpk = dominant_freq_from_psd(f, pxx, fmin=fmin, fmax=min(fmax, nyq-1e-6))

    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    crest = float(peak / rms) if rms > 0 else float("nan")

    return {
        "name": label,
        "fs": float(fs),
        "nyq": float(nyq),
        "duration_s": float(t[-1]-t[0]),
        "dominant_hz": float(fpk),
        "rpm_obs": float(fpk*60),
        "rpm_mirror": float((fs - fpk)*60),
        "rms": rms,
        "crest": crest,
        "t": t, "y": y,
        "f_psd": f, "pxx": pxx,
    }

def summarize_runs(runs):
    dom = np.array([r["dominant_hz"] for r in runs], dtype=float)
    rms = np.array([r["rms"] for r in runs], dtype=float)
    return {
        "n_runs": len(runs),
        "dominant_hz_mean": float(dom.mean()) if len(dom) else np.nan,
        "dominant_hz_std": float(dom.std(ddof=1)) if len(dom) > 1 else 0.0,
        "repeatability_dom_hz_delta": float(dom.max()-dom.min()) if len(dom) > 1 else 0.0,
        "rms_mean": float(rms.mean()) if len(rms) else np.nan,
        "rms_std": float(rms.std(ddof=1)) if len(rms) > 1 else 0.0,
        "repeatability_rms_cv": float(rms.std(ddof=1)/rms.mean()) if len(rms) > 1 and rms.mean() > 0 else 0.0,
    }

def average_psd(runs, fmax=300.0, n_points=1200):
    if not runs:
        return None, None
    f_common = np.linspace(0.0, fmax, n_points)
    stack = []
    for r in runs:
        f = r["f_psd"]; p = r["pxx"]
        if len(f) < 2:
            continue
        stack.append(np.interp(f_common, f, p, left=np.nan, right=np.nan))
    if not stack:
        return None, None
    P = np.vstack(stack)
    return f_common, np.nanmean(P, axis=0)

# ---------- Acoustics ----------
def detect_audio_type(df: pd.DataFrame):
    tcol = _find_col(df.columns, ["time"], ["s"]) or _find_col(df.columns, ["time"])
    fcol = _find_col(df.columns, ["frequency"], ["hz"]) or _find_col(df.columns, ["frequency"])
    if tcol and fcol:
        return "freq_history", tcol, fcol, None

    fcol = _find_col(df.columns, ["frequency"], ["hz"]) or _find_col(df.columns, ["frequency"])
    acol = _find_col(df.columns, ["amplitude"]) or _find_col(df.columns, ["level"]) or _find_col(df.columns, ["power"])
    if fcol and acol:
        return "spectrum", None, fcol, acol

    raise ValueError(f"Could not detect audio CSV type. Columns: {list(df.columns)}")

def _acoustics_freq_history_metrics(t: np.ndarray, f: np.ndarray):
    mask = np.isfinite(t) & np.isfinite(f)
    t = t[mask]
    f = f[mask]
    if len(f) < 5:
        return {
            "tone_mean_hz": float(np.nanmean(f)) if len(f) else np.nan,
            "tone_std_hz": 0.0,
            "tone_iqr_hz": 0.0,
            "drift_hz_per_s": 0.0,
            "dropout_pct": 0.0,
        }

    tone_mean = float(np.mean(f))
    tone_std = float(np.std(f, ddof=1)) if len(f) > 1 else 0.0
    tone_iqr = _iqr(f)

    tt = t - t[0]
    drift = float(np.polyfit(tt, f, 1)[0]) if len(tt) > 1 else 0.0

    # Dropout heuristic: deep downward spikes when tracker loses lock
    med = float(np.median(f))
    thr = med - max(150.0, 0.30 * med)   # robust threshold
    dropout_pct = float(np.mean(f < thr) * 100.0)

    return {
        "tone_mean_hz": tone_mean,
        "tone_std_hz": tone_std,
        "tone_iqr_hz": float(tone_iqr),
        "drift_hz_per_s": drift,
        "dropout_pct": dropout_pct,
    }

def analyze_audio_file(uploaded_file, label):
    df = _read_csv(uploaded_file)
    mode, tcol, fcol, acol = detect_audio_type(df)

    if mode == "freq_history":
        t = df[tcol].to_numpy(dtype=float)
        f = df[fcol].to_numpy(dtype=float)
        mets = _acoustics_freq_history_metrics(t, f)
        return {"name": label, "mode": "freq_history", "t_series": t, "f_series": f, **mets}

    # spectrum
    freq = df[fcol].to_numpy(dtype=float)
    amp = df[acol].to_numpy(dtype=float)
    i = int(np.argmax(amp))
    return {"name": label, "mode": "spectrum", "tone_hz": float(freq[i]), "tone_amp": float(amp[i]), "freq": freq, "amp": amp}

def summarize_audio_runs(audio_runs):
    if not audio_runs:
        return None
    modes = {r["mode"] for r in audio_runs}
    if len(modes) != 1:
        return {"mode": "mixed", "n_runs": len(audio_runs)}
    mode = list(modes)[0]

    if mode == "freq_history":
        mean = np.array([r["tone_mean_hz"] for r in audio_runs], dtype=float)
        std = np.array([r["tone_std_hz"] for r in audio_runs], dtype=float)
        iqr = np.array([r["tone_iqr_hz"] for r in audio_runs], dtype=float)
        drift = np.array([r["drift_hz_per_s"] for r in audio_runs], dtype=float)
        dropout = np.array([r["dropout_pct"] for r in audio_runs], dtype=float)

        def _m(x): return float(np.nanmean(x)) if len(x) else np.nan

        s_mean = _m(std)
        d_mean = _m(dropout)

        # Simple stability label (works well for phone tone-tracker)
        if (s_mean < 60.0) and (d_mean < 10.0):
            label = "Stable"
        elif (s_mean < 120.0) and (d_mean < 25.0):
            label = "Moderate"
        else:
            label = "Unstable"

        return {
            "mode": mode,
            "n_runs": len(audio_runs),
            "tone_mean_hz_mean": _m(mean),
            "tone_mean_hz_std": float(np.nanstd(mean, ddof=1)) if len(mean) > 1 else 0.0,
            "tone_std_hz_mean": s_mean,
            "tone_iqr_hz_mean": _m(iqr),
            "drift_hz_per_s_mean": _m(drift),
            "dropout_pct_mean": d_mean,
            "stability": label,
        }

    # spectrum
    tones = np.array([r["tone_hz"] for r in audio_runs], dtype=float)
    def _m(x): return float(np.nanmean(x)) if len(x) else np.nan
    return {"mode": mode, "n_runs": len(audio_runs), "dominant_tone_hz_mean": _m(tones), "dominant_tone_hz_std": float(np.nanstd(tones, ddof=1)) if len(tones) > 1 else 0.0}

# ---------- Plot helpers ----------
def plotly_chart(fig):
    st.plotly_chart(fig, width="stretch")

def mpl_chart(fig):
    st.pyplot(fig, clear_figure=True)

def plot_psd_overlay_runs(runs, title):
    if PLOTLY_OK:
        fig = go.Figure()
        for r in runs:
            fig.add_trace(go.Scatter(x=r["f_psd"], y=r["pxx"], mode="lines", name=r["name"]))
        fig.update_layout(title=title, xaxis_title="Frequency (Hz)", yaxis_title="PSD", yaxis_type="log")
        plotly_chart(fig)
    else:
        fig = plt.figure(figsize=(10,5))
        for r in runs:
            plt.semilogy(r["f_psd"], r["pxx"], label=r["name"], alpha=0.85)
        plt.xlim(0, 300); plt.grid(True, alpha=0.3)
        plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD"); plt.title(title)
        plt.legend(fontsize=9)
        mpl_chart(fig)

def plot_psd_low_high_avg(vib_low, vib_high, title):
    fL, pL = average_psd(vib_low)
    fH, pH = average_psd(vib_high)
    if fL is None or fH is None:
        st.info("Not enough data to compute averaged PSD.")
        return

    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fL, y=pL, mode="lines", name="Low (avg)"))
        fig.add_trace(go.Scatter(x=fH, y=pH, mode="lines", name="High (avg)"))
        fig.update_layout(title=title, xaxis_title="Frequency (Hz)", yaxis_title="PSD", yaxis_type="log")
        plotly_chart(fig)
    else:
        fig = plt.figure(figsize=(10,5))
        plt.semilogy(fL, pL, label="Low (avg)")
        plt.semilogy(fH, pH, label="High (avg)")
        plt.xlim(0, 300); plt.grid(True, alpha=0.3)
        plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD"); plt.title(title)
        plt.legend(fontsize=9)
        mpl_chart(fig)

def plot_time_overlay_runs(runs, title):
    if PLOTLY_OK:
        fig = go.Figure()
        for r in runs:
            t = r["t"] - r["t"][0]
            fig.add_trace(go.Scatter(x=t, y=r["y"], mode="lines", name=r["name"]))
        fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Filtered vibration (a.u.)")
        plotly_chart(fig)
    else:
        fig = plt.figure(figsize=(10,4))
        for r in runs:
            t = r["t"] - r["t"][0]
            plt.plot(t, r["y"], label=r["name"], alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.xlabel("Time (s)"); plt.ylabel("Filtered vibration (a.u.)"); plt.title(title)
        plt.legend(fontsize=9)
        mpl_chart(fig)

def plot_spectrogram(run, title):
    f, t, SdB = spectrogram_db(run["y"], run["fs"])
    m = f <= 300
    f = f[m]; SdB = SdB[m, :]

    if PLOTLY_OK:
        fig = go.Figure(data=go.Heatmap(x=t, y=f, z=SdB, colorbar=dict(title="dB (rel.)")))
        fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
        plotly_chart(fig)
    else:
        fig = plt.figure(figsize=(10,4.8))
        plt.pcolormesh(t, f, SdB, shading="auto")
        plt.ylim(0, 300)
        plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz)"); plt.title(title)
        plt.colorbar(label="dB (relative)")
        mpl_chart(fig)

def plot_freq_history_overlay(audio_runs, title):
    if PLOTLY_OK:
        fig = go.Figure()
        for r in audio_runs:
            fig.add_trace(go.Scatter(x=r["t_series"], y=r["f_series"], mode="lines", name=r["name"]))
        fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
        plotly_chart(fig)
    else:
        fig = plt.figure(figsize=(10,4))
        for r in audio_runs:
            plt.plot(r["t_series"], r["f_series"], alpha=0.8, label=r["name"])
        plt.grid(True, alpha=0.3)
        plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz)"); plt.title(title)
        plt.legend(fontsize=9)
        mpl_chart(fig)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("View")
    persona = st.radio("Who are you?", ["Stakeholder", "Engineer"], index=0)

    st.markdown("---")
    st.subheader("Upload (Vibration)")
    vib_low_files = st.file_uploader("Low speed CSVs (2+ runs recommended)", type=["csv"], accept_multiple_files=True)
    vib_high_files = st.file_uploader("High speed CSVs (2+ runs recommended)", type=["csv"], accept_multiple_files=True)

    st.markdown("---")
    st.subheader("Upload (Acoustics) — optional")
    audio_low_files = st.file_uploader("Low speed audio (Frequency history recommended)", type=["csv"], accept_multiple_files=True)
    audio_high_files = st.file_uploader("High speed audio (Frequency history recommended)", type=["csv"], accept_multiple_files=True)

    st.markdown("---")
    if persona == "Engineer":
        st.subheader("Advanced (Engineer)")
        hp = st.slider("Vibration high-pass filter (Hz)", 0.0, 30.0, 5.0, 0.5)
        lp = st.slider("Vibration low-pass filter (Hz)", 30.0, 250.0, 200.0, 5.0)
        fmin = st.slider("Vibration peak search min (Hz)", 0, 50, 10, 1)
        fmax = st.slider("Vibration peak search max (Hz)", 50, 250, 200, 5)
    else:
        hp, lp, fmin, fmax = 5.0, 200.0, 10, 200
        st.caption("Stakeholder view uses defaults; switch to Engineer for advanced settings.")

    st.markdown("---")
    st.caption("Tip: `pip install plotly` for zoom/hover charts.")

# ---------- Compute analyses (raw) ----------
vib_low_all, vib_high_all, vib_errors = [], [], []
for i, uf in enumerate(vib_low_files or []):
    try:
        vib_low_all.append(analyze_vibration_file(uf, hp, lp, fmin, fmax, label=_safe_label(uf.name, i)))
    except Exception as e:
        vib_errors.append((_safe_label(uf.name, i), str(e)))

for i, uf in enumerate(vib_high_files or []):
    try:
        vib_high_all.append(analyze_vibration_file(uf, hp, lp, fmin, fmax, label=_safe_label(uf.name, i)))
    except Exception as e:
        vib_errors.append((_safe_label(uf.name, i), str(e)))

audio_low_all, audio_high_all, audio_errors = [], [], []
for i, uf in enumerate(audio_low_files or []):
    try:
        audio_low_all.append(analyze_audio_file(uf, label=_safe_label(uf.name, i)))
    except Exception as e:
        audio_errors.append((_safe_label(uf.name, i), str(e)))

for i, uf in enumerate(audio_high_files or []):
    try:
        audio_high_all.append(analyze_audio_file(uf, label=_safe_label(uf.name, i)))
    except Exception as e:
        audio_errors.append((_safe_label(uf.name, i), str(e)))

# ---------- Run selection (Engineer only) ----------
vib_low, vib_high = vib_low_all, vib_high_all
audio_low, audio_high = audio_low_all, audio_high_all

if persona == "Engineer" and (vib_low_all or vib_high_all or audio_low_all or audio_high_all):
    with st.sidebar:
        st.markdown("---")
        st.subheader("Run selection (Engineer)")
        if vib_low_all:
            opt = [r["name"] for r in vib_low_all]
            sel = st.multiselect("Include vibration LOW runs", opt, default=opt)
            vib_low = [r for r in vib_low_all if r["name"] in sel]
        if vib_high_all:
            opt = [r["name"] for r in vib_high_all]
            sel = st.multiselect("Include vibration HIGH runs", opt, default=opt)
            vib_high = [r for r in vib_high_all if r["name"] in sel]
        if audio_low_all:
            opt = [r["name"] for r in audio_low_all]
            sel = st.multiselect("Include audio LOW runs", opt, default=opt)
            audio_low = [r for r in audio_low_all if r["name"] in sel]
        if audio_high_all:
            opt = [r["name"] for r in audio_high_all]
            sel = st.multiselect("Include audio HIGH runs", opt, default=opt)
            audio_high = [r for r in audio_high_all if r["name"] in sel]

vib_ready = bool(vib_low and vib_high)
audio_ready = bool(audio_low and audio_high)

# ---------- Tabs ----------
tabs = st.tabs(["Overview", "Vibration", "Acoustics", "Combined"])

# ---------- Overview ----------
with tabs[0]:
    st.markdown(
        f"<div class='card'>"
        f"<span class='badge'>Portfolio demo</span> "
        f"<span class='badge2'>Updated {datetime.now().strftime('%Y-%m-%d')}</span>"
        f"<hr class='hr'/>"
        f"<div class='small'>Upload Phyphox CSVs → compare low vs high speed vibration & acoustics → check repeatability → export stakeholder-friendly summaries.</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    if vib_errors:
        st.warning("Some vibration files could not be parsed:")
        for n, e in vib_errors:
            st.write(f"- {n}: {e}")

    if vib_ready:
        low_sum = summarize_runs(vib_low)
        high_sum = summarize_runs(vib_high)
        stronger = "High speed" if high_sum["rms_mean"] > low_sum["rms_mean"] else "Low speed"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Low RMS", f"{low_sum['rms_mean']:.2f}")
        c2.metric("High RMS", f"{high_sum['rms_mean']:.2f}")
        c3.metric("Dom freq (Low)", f"{low_sum['dominant_hz_mean']:.1f} Hz")
        c4.metric("Dom freq (High)", f"{high_sum['dominant_hz_mean']:.1f} Hz")

        st.success(f"Headline: **{stronger}** shows higher vibration strength (RMS) in these measurements.")

        if low_sum["n_runs"] < 2 or high_sum["n_runs"] < 2:
            st.warning("Repeatability needs ≥2 runs per speed. Upload/select at least two low-speed and two high-speed runs.")

        metrics_df = pd.DataFrame([{"Speed": "Low", **low_sum}, {"Speed": "High", **high_sum}])
        st.download_button(
            "Download vibration summary CSV",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="vibration_summary.csv",
            mime="text/csv",
            key="dl_vib_overview",
        )
    else:
        st.info("Upload/select at least one low-speed and one high-speed vibration run.")

    if audio_errors:
        st.warning("Some audio files could not be parsed:")
        for n, e in audio_errors:
            st.write(f"- {n}: {e}")

    if audio_ready:
        modes = {r["mode"] for r in audio_low + audio_high}
        if len(modes) == 1 and list(modes)[0] == "freq_history":
            low_a = summarize_audio_runs(audio_low)
            high_a = summarize_audio_runs(audio_high)
            st.write(f"Acoustics (tone mean): **Low ≈ {low_a['tone_mean_hz_mean']:.1f} Hz**, **High ≈ {high_a['tone_mean_hz_mean']:.1f} Hz**")
            st.write(f"Stability: **Low = {low_a['stability']}**, **High = {high_a['stability']}** (tone variability + dropout).")
        elif len(modes) > 1:
            st.error("Audio uploads mix different types. Upload only Frequency history OR only Spectrum.")
        else:
            st.info("Acoustics in spectrum mode detected (frequency peaks).")

# ---------- Vibration ----------
with tabs[1]:
    st.subheader("Vibration")

    if not vib_ready:
        st.info("Upload/select low + high vibration CSVs.")
    else:
        low_sum = summarize_runs(vib_low)
        high_sum = summarize_runs(vib_high)
        metrics_df = pd.DataFrame([{"Speed": "Low", **low_sum}, {"Speed": "High", **high_sum}])

        if persona == "Stakeholder":
            stronger = "High speed" if high_sum["rms_mean"] > low_sum["rms_mean"] else "Low speed"
            st.markdown("<div class='card'><h3 style='margin:0;'>Stakeholder summary</h3><div class='small'>One averaged PSD curve per speed.</div></div>", unsafe_allow_html=True)
            st.success(f"**{stronger}** shows higher vibration strength in this dataset.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Low RMS", f"{low_sum['rms_mean']:.2f}")
            m2.metric("High RMS", f"{high_sum['rms_mean']:.2f}")
            m3.metric("Repeatability Δpeak (Hz)", f"{max(low_sum['repeatability_dom_hz_delta'], high_sum['repeatability_dom_hz_delta']):.1f}")

            st.download_button(
                "Download vibration summary CSV",
                data=metrics_df.to_csv(index=False).encode("utf-8"),
                file_name="vibration_summary.csv",
                mime="text/csv",
                key="dl_vib_tab_stakeholder",
            )

            plot_psd_low_high_avg(vib_low, vib_high, "PSD (averaged): Low vs High")

        else:
            st.markdown("<div class='card'><h3 style='margin:0;'>Engineer view</h3><div class='small'>Per-run overlays + spectrogram. Use Run selection to exclude droop/unstable runs.</div></div>", unsafe_allow_html=True)
            st.dataframe(metrics_df, width="stretch")

            st.download_button(
                "Download vibration summary CSV",
                data=metrics_df.to_csv(index=False).encode("utf-8"),
                file_name="vibration_summary.csv",
                mime="text/csv",
                key="dl_vib_tab_engineer",
            )

            c1, c2 = st.columns(2)
            with c1:
                plot_time_overlay_runs(vib_low, "Low speed — filtered vibration (per-run overlay)")
            with c2:
                plot_time_overlay_runs(vib_high, "High speed — filtered vibration (per-run overlay)")

            c3, c4 = st.columns(2)
            with c3:
                plot_psd_overlay_runs(vib_low, "Low speed — PSD (per run)")
            with c4:
                plot_psd_overlay_runs(vib_high, "High speed — PSD (per run)")

            st.subheader("Spectrogram (example run)")
            s1, s2 = st.columns(2)
            with s1:
                plot_spectrogram(vib_low[0], f"Low speed — spectrogram ({vib_low[0]['name']})")
            with s2:
                plot_spectrogram(vib_high[0], f"High speed — spectrogram ({vib_high[0]['name']})")

# ---------- Acoustics ----------
with tabs[2]:
    st.subheader("Acoustics")

    if not audio_ready:
        st.info("Upload low + high acoustics CSVs (Frequency history recommended).")
    else:
        modes = {r["mode"] for r in audio_low + audio_high}
        if len(modes) > 1:
            st.error("Mixed audio file types detected. Upload only one type.")
        else:
            mode = list(modes)[0]

            if mode == "freq_history":
                low_s = summarize_audio_runs(audio_low)
                high_s = summarize_audio_runs(audio_high)

                aud_df = pd.DataFrame(
                    [{"Speed": "Low", **low_s}, {"Speed": "High", **high_s}]
                )[[
                    "Speed", "n_runs",
                    "tone_mean_hz_mean", "tone_mean_hz_std",
                    "tone_std_hz_mean", "tone_iqr_hz_mean",
                    "drift_hz_per_s_mean", "dropout_pct_mean",
                    "stability"
                ]]

                st.dataframe(aud_df, width="stretch")

                st.download_button(
                    "Download acoustics summary CSV",
                    data=aud_df.to_csv(index=False).encode("utf-8"),
                    file_name="audio_summary.csv",
                    mime="text/csv",
                    key="dl_audio_tab_fh",
                )

                c1, c2 = st.columns(2)
                with c1:
                    plot_freq_history_overlay(audio_low, "Low speed — tone tracking (per run)")
                with c2:
                    plot_freq_history_overlay(audio_high, "High speed — tone tracking (per run)")

                if persona == "Engineer":
                    with st.expander("Per-run acoustics KPIs (Engineer)"):
                        per_run = pd.DataFrame(
                            [{"Speed": "Low", **r} for r in audio_low] +
                            [{"Speed": "High", **r} for r in audio_high]
                        )[["Speed", "name", "tone_mean_hz", "tone_std_hz", "tone_iqr_hz", "drift_hz_per_s", "dropout_pct"]]
                        st.dataframe(per_run, width="stretch")

                with st.expander("Plain English"):
                    st.markdown("""
- **Tone (Hz)** is like the pitch you hear. Higher speed usually shifts tone upward.
- **Tone stability (std/IQR)** shows how steady the operating point is across the run.
- **Dropout rate** is a proxy for the tracker losing lock (often seen as sharp dips).
- This is a cross-check alongside vibration to confirm operating-point changes.
""")
            else:
                st.info("Spectrum mode is supported, but Frequency history is recommended for stability/drift.")

# ---------- Combined ----------
with tabs[3]:
    st.subheader("Combined")

    if not (vib_ready and audio_ready):
        st.info("Upload/select both vibration and acoustics (low+high).")
    else:
        low_v = summarize_runs(vib_low)
        high_v = summarize_runs(vib_high)

        modes = {r["mode"] for r in audio_low + audio_high}
        if len(modes) == 1 and list(modes)[0] == "freq_history":
            low_a = summarize_audio_runs(audio_low)
            high_a = summarize_audio_runs(audio_high)

            combo = pd.DataFrame([
                {
                    "Speed": "Low",
                    "Vibration dom (Hz)": low_v["dominant_hz_mean"],
                    "Vibration RMS": low_v["rms_mean"],
                    "Audio tone (Hz)": low_a["tone_mean_hz_mean"],
                    "Audio stability": low_a["stability"],
                },
                {
                    "Speed": "High",
                    "Vibration dom (Hz)": high_v["dominant_hz_mean"],
                    "Vibration RMS": high_v["rms_mean"],
                    "Audio tone (Hz)": high_a["tone_mean_hz_mean"],
                    "Audio stability": high_a["stability"],
                },
            ])
        else:
            combo = pd.DataFrame([
                {"Speed": "Low", "Vibration dom (Hz)": low_v["dominant_hz_mean"], "Vibration RMS": low_v["rms_mean"]},
                {"Speed": "High", "Vibration dom (Hz)": high_v["dominant_hz_mean"], "Vibration RMS": high_v["rms_mean"]},
            ])

        st.dataframe(combo, width="stretch")

        st.download_button(
            "Download combined summary CSV",
            data=combo.to_csv(index=False).encode("utf-8"),
            file_name="combined_summary.csv",
            mime="text/csv",
            key="dl_combined",
        )

st.markdown("---")
st.caption("Phone sensors are not calibrated industrial instruments; this dashboard demonstrates methodology: measurement → repeatability → signature comparison → cross-check → clear reporting.")