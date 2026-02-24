# app_streamlit.py
# Hair Dryer Condition Monitoring — Vibration + Acoustics (Phyphox)
#
# Install:
#   pip install streamlit numpy pandas scipy matplotlib
#   pip install plotly   # optional (recommended) for zoom/hover plots
#
# Run:
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

# ---------- Header (hero + sidebar logo) ----------
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

def analyze_vibration_file(uploaded_file, hp, lp, fmin, fmax):
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
        "name": uploaded_file.name,
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
    # Frequency history: time + frequency
    tcol = _find_col(df.columns, ["time"], ["s"]) or _find_col(df.columns, ["time"])
    fcol = _find_col(df.columns, ["frequency"], ["hz"]) or _find_col(df.columns, ["frequency"])
    if tcol and fcol:
        return "freq_history", tcol, fcol, None

    # Spectrum: frequency + amplitude/level/power
    fcol = _find_col(df.columns, ["frequency"], ["hz"]) or _find_col(df.columns, ["frequency"])
    acol = _find_col(df.columns, ["amplitude"]) or _find_col(df.columns, ["level"]) or _find_col(df.columns, ["power"])
    if fcol and acol:
        return "spectrum", None, fcol, acol

    raise ValueError(f"Could not detect audio CSV type. Columns: {list(df.columns)}")

def analyze_audio_file(uploaded_file):
    df = _read_csv(uploaded_file)
    mode, tcol, fcol, acol = detect_audio_type(df)

    if mode == "freq_history":
        t = df[tcol].to_numpy(dtype=float)
        f = df[fcol].to_numpy(dtype=float)
        f2 = f[np.isfinite(f)]
        return {
            "name": uploaded_file.name,
            "mode": "freq_history",
            "f_mean": float(np.mean(f2)) if len(f2) else np.nan,
            "f_std": float(np.std(f2, ddof=1)) if len(f2) > 1 else 0.0,
            "t_series": t, "f_series": f,
        }

    freq = df[fcol].to_numpy(dtype=float)
    amp = df[acol].to_numpy(dtype=float)
    i = int(np.argmax(amp))
    return {
        "name": uploaded_file.name,
        "mode": "spectrum",
        "tone_hz": float(freq[i]),
        "tone_amp": float(amp[i]),
        "freq": freq, "amp": amp,
    }

# ---------- Rendering helpers ----------
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

def plot_spectrum_overlay(audio_runs, title):
    if PLOTLY_OK:
        fig = go.Figure()
        for r in audio_runs:
            fig.add_trace(go.Scatter(x=r["freq"], y=r["amp"], mode="lines", name=r["name"]))
        fig.update_layout(title=title, xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
        plotly_chart(fig)
    else:
        fig = plt.figure(figsize=(10,4))
        for r in audio_runs:
            plt.plot(r["freq"], r["amp"], alpha=0.8, label=r["name"])
        plt.grid(True, alpha=0.3)
        plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude"); plt.title(title)
        plt.legend(fontsize=9)
        mpl_chart(fig)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("View")
    persona = st.radio("Who are you?", ["Stakeholder", "Engineer"], index=0)

    st.markdown("---")
    st.subheader("Upload (Vibration)")
    vib_low_files = st.file_uploader("Low speed CSVs (2 runs recommended)", type=["csv"], accept_multiple_files=True)
    vib_high_files = st.file_uploader("High speed CSVs (2 runs recommended)", type=["csv"], accept_multiple_files=True)

    st.markdown("---")
    st.subheader("Upload (Acoustics) — optional")
    audio_low_files = st.file_uploader("Low speed audio CSVs", type=["csv"], accept_multiple_files=True)
    audio_high_files = st.file_uploader("High speed audio CSVs", type=["csv"], accept_multiple_files=True)

    st.markdown("---")
    if persona == "Engineer":
        st.subheader("Advanced (Engineer)")
        hp = st.slider("High-pass filter (Hz)", 0.0, 30.0, 5.0, 0.5)
        lp = st.slider("Low-pass filter (Hz)", 30.0, 250.0, 200.0, 5.0)
        fmin = st.slider("Peak search min (Hz)", 0, 50, 10, 1)
        fmax = st.slider("Peak search max (Hz)", 50, 250, 200, 5)
    else:
        hp, lp, fmin, fmax = 5.0, 200.0, 10, 200
        st.caption("Stakeholder view uses defaults; switch to Engineer for advanced settings.")

    st.markdown("---")
    st.caption("Tip: `pip install plotly` for zoom/hover charts.")

# ---------- Compute analyses ----------
vib_low, vib_high, vib_errors = [], [], []
if vib_low_files or vib_high_files:
    for uf in vib_low_files:
        try:
            vib_low.append(analyze_vibration_file(uf, hp, lp, fmin, fmax))
        except Exception as e:
            vib_errors.append((uf.name, str(e)))
    for uf in vib_high_files:
        try:
            vib_high.append(analyze_vibration_file(uf, hp, lp, fmin, fmax))
        except Exception as e:
            vib_errors.append((uf.name, str(e)))

vib_ready = bool(vib_low and vib_high)

audio_low, audio_high, audio_errors = [], [], []
if audio_low_files or audio_high_files:
    for uf in audio_low_files:
        try:
            audio_low.append(analyze_audio_file(uf))
        except Exception as e:
            audio_errors.append((uf.name, str(e)))
    for uf in audio_high_files:
        try:
            audio_high.append(analyze_audio_file(uf))
        except Exception as e:
            audio_errors.append((uf.name, str(e)))

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
        f"<div class='small'>Upload Phyphox CSVs → compare low vs high speed vibration & acoustics → download stakeholder-friendly summaries.</div>"
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
        c3.metric("Dominant freq (Low)", f"{low_sum['dominant_hz_mean']:.1f} Hz")
        c4.metric("Dominant freq (High)", f"{high_sum['dominant_hz_mean']:.1f} Hz")

        st.success(f"Headline: **{stronger}** vibrates more (based on RMS).")

        if low_sum["n_runs"] < 2 or high_sum["n_runs"] < 2:
            st.warning("Repeatability needs ≥2 runs per speed. Upload two low-speed and two high-speed CSVs for a proper repeatability check.")

        metrics_df = pd.DataFrame([
            {"Speed": "Low", **low_sum},
            {"Speed": "High", **high_sum},
        ])

        st.download_button(
            "Download vibration summary CSV",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="vibration_summary.csv",
            mime="text/csv",
            key="dl_vib_overview",
        )
    else:
        st.info("Upload at least one low-speed and one high-speed vibration CSV to populate the overview.")

    if audio_errors:
        st.warning("Some audio files could not be parsed:")
        for n, e in audio_errors:
            st.write(f"- {n}: {e}")

    if audio_ready:
        modes = {r["mode"] for r in audio_low + audio_high}
        if len(modes) == 1:
            mode = list(modes)[0]
            if mode == "freq_history":
                low_mean = float(np.mean([r["f_mean"] for r in audio_low]))
                high_mean = float(np.mean([r["f_mean"] for r in audio_high]))
                st.write(f"Acoustics (tone mean): **Low ≈ {low_mean:.1f} Hz**, **High ≈ {high_mean:.1f} Hz**")
            else:
                low_tone = float(np.mean([r["tone_hz"] for r in audio_low]))
                high_tone = float(np.mean([r["tone_hz"] for r in audio_high]))
                st.write(f"Acoustics (dominant tone): **Low ≈ {low_tone:.1f} Hz**, **High ≈ {high_tone:.1f} Hz**")
        else:
            st.error("Audio uploads mix different types (Frequency history + Spectrum). Upload only one type at a time.")

# ---------- Vibration tab ----------
with tabs[1]:
    st.subheader("Vibration")

    if not vib_ready:
        st.info("Upload low + high vibration CSVs in the sidebar.")
    else:
        low_sum = summarize_runs(vib_low)
        high_sum = summarize_runs(vib_high)
        stronger = "High speed" if high_sum["rms_mean"] > low_sum["rms_mean"] else "Low speed"

        if low_sum["n_runs"] < 2 or high_sum["n_runs"] < 2:
            st.warning("Repeatability needs ≥2 runs per speed. Upload two low-speed and two high-speed runs.")

        metrics_df = pd.DataFrame([
            {"Speed": "Low", **low_sum},
            {"Speed": "High", **high_sum},
        ])

        if persona == "Stakeholder":
            st.markdown(
                "<div class='card'><h3 style='margin:0;'>Stakeholder summary</h3>"
                "<div class='small'>Simple view: one averaged PSD curve per speed.</div></div>",
                unsafe_allow_html=True
            )
            st.success(f"**{stronger}** shows higher vibration strength in this dataset.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Low vibration (RMS)", f"{low_sum['rms_mean']:.2f}")
            m2.metric("High vibration (RMS)", f"{high_sum['rms_mean']:.2f}")
            m3.metric("Repeatability (Δpeak, Hz)", f"{max(low_sum['repeatability_dom_hz_delta'], high_sum['repeatability_dom_hz_delta']):.1f}")

            st.download_button(
                "Download vibration summary CSV",
                data=metrics_df.to_csv(index=False).encode("utf-8"),
                file_name="vibration_summary.csv",
                mime="text/csv",
                key="dl_vib_tab_stakeholder",
            )

            plot_psd_low_high_avg(vib_low, vib_high, "PSD (averaged): Low vs High")

            with st.expander("What do these plots mean? (plain English)"):
                st.markdown("""
- **PSD plot** shows which frequencies contain most of the vibration energy (a machine “signature”).
- If the curve changes between low and high speed, it indicates the operating condition changed.
- Two runs that look similar means the measurement is more trustworthy (**repeatability**).
""")
        else:
            st.markdown(
                "<div class='card'><h3 style='margin:0;'>Engineer view</h3>"
                "<div class='small'>Per-run overlays + time–frequency plots.</div></div>",
                unsafe_allow_html=True
            )

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

            st.caption("Note: phone sampling limits can cause aliasing; industrial tests use calibrated sensors + tach/keyphasor.")

# ---------- Acoustics tab ----------
with tabs[2]:
    st.subheader("Acoustics")

    if not audio_ready:
        st.info("Upload low + high acoustics CSVs (Frequency history OR Audio spectrum).")
    else:
        modes = {r["mode"] for r in audio_low + audio_high}
        if len(modes) > 1:
            st.error("Mixed audio file types detected (Frequency history + Spectrum). Upload only one type at a time.")
        else:
            mode = list(modes)[0]

            if mode == "freq_history":
                low_mean = float(np.mean([r["f_mean"] for r in audio_low]))
                high_mean = float(np.mean([r["f_mean"] for r in audio_high]))

                aud_df = pd.DataFrame([
                    {"Speed": "Low", "Tone mean (Hz)": low_mean, "Runs": len(audio_low)},
                    {"Speed": "High", "Tone mean (Hz)": high_mean, "Runs": len(audio_high)},
                ])

                if persona == "Stakeholder":
                    st.success("This shows how the main pitch (tone) shifts between speeds.")
                else:
                    st.info("Tone tracking over time (frequency history).")

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
                    plot_freq_history_overlay(audio_low, "Low speed — tone tracking")
                with c2:
                    plot_freq_history_overlay(audio_high, "High speed — tone tracking")

            else:
                low_tone = float(np.mean([r["tone_hz"] for r in audio_low]))
                high_tone = float(np.mean([r["tone_hz"] for r in audio_high]))

                aud_df = pd.DataFrame([
                    {"Speed": "Low", "Dominant tone (Hz)": low_tone, "Runs": len(audio_low)},
                    {"Speed": "High", "Dominant tone (Hz)": high_tone, "Runs": len(audio_high)},
                ])

                if persona == "Stakeholder":
                    st.success("This compares the strongest tone in the sound spectrum between speeds.")
                else:
                    st.info("Audio spectrum (frequency content).")

                st.dataframe(aud_df, width="stretch")

                st.download_button(
                    "Download acoustics summary CSV",
                    data=aud_df.to_csv(index=False).encode("utf-8"),
                    file_name="audio_summary.csv",
                    mime="text/csv",
                    key="dl_audio_tab_spec",
                )

                c1, c2 = st.columns(2)
                with c1:
                    plot_spectrum_overlay(audio_low, "Low speed — spectrum (per run)")
                with c2:
                    plot_spectrum_overlay(audio_high, "High speed — spectrum (per run)")

            if persona == "Stakeholder":
                with st.expander("Plain English"):
                    st.markdown("""
- The **tone** is the pitch you hear.
- If the tone shifts between low and high speed, it usually means the fan/motor speed changed.
- Engineers compare tone + vibration signatures to troubleshoot discrepancies.
""")

# ---------- Combined tab ----------
with tabs[3]:
    st.subheader("Combined")

    if not (vib_ready and audio_ready):
        st.info("Upload both vibration and acoustics (low+high) to see a combined summary.")
    else:
        modes = {r["mode"] for r in audio_low + audio_high}
        if len(modes) > 1:
            st.error("Mixed audio file types detected (Frequency history + Spectrum). Upload only one type at a time.")
        else:
            low_sum = summarize_runs(vib_low)
            high_sum = summarize_runs(vib_high)

            mode = list(modes)[0]
            if mode == "freq_history":
                low_a = float(np.mean([r["f_mean"] for r in audio_low]))
                high_a = float(np.mean([r["f_mean"] for r in audio_high]))
            else:
                low_a = float(np.mean([r["tone_hz"] for r in audio_low]))
                high_a = float(np.mean([r["tone_hz"] for r in audio_high]))

            combo = pd.DataFrame([
                {"Speed": "Low", "Vibration dominant (Hz)": low_sum["dominant_hz_mean"], "Audio tone (Hz)": low_a},
                {"Speed": "High", "Vibration dominant (Hz)": high_sum["dominant_hz_mean"], "Audio tone (Hz)": high_a},
            ])

            st.dataframe(combo, width="stretch")

            st.download_button(
                "Download combined summary CSV",
                data=combo.to_csv(index=False).encode("utf-8"),
                file_name="combined_summary.csv",
                mime="text/csv",
                key="dl_combined",
            )

            if persona == "Stakeholder":
                st.success("If vibration and sound shift consistently between speeds, it supports that the change is operating-condition related.")
            else:
                st.info("For true order tracking you need a tach/keyphasor; here we focus on comparative signatures + repeatability.")

st.markdown("---")
st.caption("Phone sensors are not calibrated industrial instruments; this dashboard demonstrates methodology: measurement → repeatability → signature comparison → clear reporting.")